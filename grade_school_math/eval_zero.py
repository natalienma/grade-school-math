import os
import re
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Dict

import torch
import torch.distributed as dist
import torch.utils.data.distributed
import transformers
from torch.cuda.amp import autocast, GradScaler
from torch.distributed.checkpoint import load_state_dict, FileSystemReader
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy, FullStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
)
from torch.distributed.fsdp.wrap import _module_wrap_policy
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import AdamW, AutoConfig
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import get_scheduler
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.trainer_pt_utils import LabelSmoother

from dataset import read_jsonl

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default="train", metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default="test", metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False
    loss_on_prefix: bool = True
    padding_left: bool = False
    eval_padding_left: bool = False
    data_debug: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    overwrite_output_dir: bool = field(default=False)


class GSMDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, examples, loss_on_prefix=True):
        self.examples = examples
        self.q = [ex["question"] for ex in self.examples]
        self.qa = [ex["question"] + ex["answer"] for ex in self.examples]
        self.q_ids = tokenizer(self.q,
                               return_tensors="pt",
                               padding="max_length",
                               max_length=tokenizer.model_max_length,
                               truncation=True).input_ids
        self.input_ids = tokenizer(self.qa,
                                   return_tensors="pt",
                                   padding="max_length",
                                   max_length=tokenizer.model_max_length,
                                   truncation=True).input_ids
        self.loss_on_prefix = loss_on_prefix
        self.targets = self.input_ids.clone()
        self.attention_mask = self.input_ids.ne(tokenizer.pad_token_id)
        self.q_attention_mask = self.q_ids.ne(tokenizer.pad_token_id)
        self.max_len = max(
            [
                len(self.input_ids[i])
                for i in range(len(self.examples))
            ]
        )
        print(f"input_ids: {self.input_ids.shape}, mask: {self.attention_mask.shape}, targets: {self.targets.shape}")
        print(f"max tokens: {self.max_len}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return dict(input_ids=self.input_ids[idx],
                    attention_mask=self.attention_mask[idx],
                    labels=self.targets[idx],
                    examples=self.examples[idx],
                    q_ids=self.q_ids[idx],
                    q_attention_mask=self.q_attention_mask[idx])
        # return dict(input_ids=tokens, attention_mask=mask)


def extract_answer(completion):
    if completion.find('\u0000') >= 0:
        completion = completion[0:completion.find('\u0000')]
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        try:
            float(match_str)
        except BaseException:
            return INVALID_ANS
        return match_str
    else:
        return INVALID_ANS


def eval(model, eval_dataloader, tokenizer):
    generation_config = GenerationConfig(
        # temperature=0.7,
        # do_sample=False,
        # num_beams=1,
        max_new_tokens=256,
        # num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    pred_ans_list = []
    gold_ans_list = []
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            print(batch['q_ids'].shape, batch['q_attention_mask'].shape)
            batch_output = model.generate(
                input_ids=batch['q_ids'].to(model.device),
                attention_mask=batch["q_attention_mask"].to(model.device),
                generation_config=generation_config,
                return_dict_in_generate=True
            )
        outputs_string = tokenizer.batch_decode(batch_output.sequences, skip_special_tokens=True)
        for gold_ans, pred_ans in zip(batch['examples']["answer"], outputs_string):
            gold_ext = extract_answer(gold_ans)
            pred_ext = extract_answer(pred_ans)
            gold_ans_list.append(gold_ext)
            pred_ans_list.append(pred_ext)
            print("GOLD: ", gold_ext, gold_ans)
            print("PRED: ", pred_ext, pred_ans)
    
    cor = 0
    invalid = 0
    rg = range(min(len(pred_ans_list), len(gold_ans_list)))
    for i in rg:
        if pred_ans_list[i] != INVALID_ANS and abs(float(pred_ans_list[i]) - float(gold_ans_list[i])) < 1e-4:
            cor += 1
        if pred_ans_list[i] == INVALID_ANS:
            invalid += 1
    print("correct: %d, total: %d, acc: %.3f, invalid: %d" % (cor, len(list(rg)), cor / len(list(rg)), invalid))


def get_examples(split):
    path = os.path.join("data/", f"{split}.jsonl")
    examples = read_jsonl(path)
    
    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")
    
    print(f"{len(examples)} {split} examples")
    return examples


def setup():
    if not dist.is_initialized():
        dist.init_process_group("nccl")


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def custom_auto_wrap_policy(module: Module, recurse: bool, nonwrapped_numel: int, min_num_params: int = 1e8) -> bool:
    if isinstance(module, LlamaDecoderLayer):
        return True
    else:
        return False


def fsdp_main():
    global batch0
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print("rank: ", rank, "world_size: ", world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        gradient_checkpointing=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )
    train_examples = get_examples(data_args.data_path)
    sample_size = len(train_examples) // 20
    import random
    train_examples = random.sample(train_examples, sample_size)
    train_dset = GSMDataset(tokenizer,
                            train_examples,
                            loss_on_prefix=data_args.loss_on_prefix)
    train_sampler = DistributedSampler(train_dset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dset,
                              batch_size=training_args.per_device_train_batch_size,
                              sampler=train_sampler,
                              shuffle=False)
    eval_examples = get_examples(data_args.eval_data_path)[:500]
    eval_dset = GSMDataset(tokenizer,
                           eval_examples,
                           loss_on_prefix=data_args.loss_on_prefix)
    eval_sampler = DistributedSampler(eval_dset, num_replicas=world_size, rank=rank)
    eval_loader = DataLoader(eval_dset,
                             batch_size=training_args.per_device_eval_batch_size,
                             sampler=eval_sampler,
                             shuffle=False)
    fsdp_config = {
        "use_orig_params": True,
        "device_id": rank,
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "auto_wrap_policy": partial(
            _module_wrap_policy,
            module_classes=(LlamaDecoderLayer,))
    }
    model = FullyShardedDataParallel(model, **fsdp_config)
    if os.path.exists(training_args.output_dir):
        print("loading model from", training_args.output_dir)
        # reader = FileSystemReader(training_args.output_dir)
        model_state = torch.load(training_args.output_dir)
        # print("reader created", reader)
        with FullyShardedDataParallel.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            model.load_state_dict(model_state)
            print("model loaded")

    if training_args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if training_args.torch_compile:
        model = torch.compile(model)
    model.train()
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
    num_epochs = int(training_args.num_train_epochs)
    num_training_steps = num_epochs * len(train_loader)
    scaler = GradScaler()
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    pbar = tqdm(range(num_training_steps))
    idx = 0
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    for epoch in range(num_epochs):
        for batch0 in train_loader:
            batch = {k: v.to(device) for k, v in batch0.items() if k in ("input_ids", "attention_mask")}
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
                outputs = model(**batch, labels=batch0["labels"])
                loss = outputs[0]
                loss = loss / batch['input_ids'].shape[0]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            max_memory = torch.cuda.max_memory_allocated()
            pbar.set_description(
                f"train_loss: {loss.item():.5f}, epoch: {epoch}, optimizer: {optimizer}, Memory: {max_memory/1e9:.3f}G")
            pbar.update(1)
            idx += 1
            # if idx % training_args.eval_steps == 0:
            #     model.eval()
            #     eval(model, eval_loader, tokenizer)
            #     model.train()
    model.eval()
    eval(model, eval_loader, tokenizer)
    output_dir = training_args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with FullyShardedDataParallel.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()
    if rank == 0:
        print("saving model to", training_args.output_dir)
        torch.save(cpu_state, training_args.output_dir)
    cleanup()


if __name__ == "__main__":
    torch.manual_seed(42)
    
    fsdp_main()
