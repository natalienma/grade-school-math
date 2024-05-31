import contextlib
import copy
import json
import logging
import os
import re
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Dict, Sequence, Union, Any

import accelerate
import torch
import torch.distributed as dist
import torch.utils.data.distributed
import transformers
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from torch.cuda.amp import autocast, GradScaler
from torch.distributed.checkpoint import load_state_dict, FileSystemReader
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy, FullStateDictConfig, StateDictType, \
    MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn import Module
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import AdamW, AutoConfig
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import get_scheduler
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.trainer_pt_utils import LabelSmoother

from dataset import read_jsonl
import torch._dynamo

torch._dynamo.config.suppress_errors = True
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "question: {instruction}\n answer:"
    ),
}


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


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    # for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
    #     label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        with open(data_path, 'r') as f:
            lines = f.readlines()
        list_data_dict = [json.loads(line.strip()) for line in lines]
        sample_size = len(list_data_dict) // 20
        import random
        list_data_dict = random.sample(list_data_dict, sample_size)
        # logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        # print(list_data_dict[0])
        if 'instruction' in list_data_dict[0]:
            pass
        else:
            def get_input(query):
                if query.find('\n') == -1:
                    return ''
                return '\n'.join(query.split('\n')[1:])
            
            list_data_dict = [{'instruction': data['query'].split('\n')[0], 'input': get_input(data['query']),
                               'output': data['response']} for data in list_data_dict]
        # import ipdb; ipdb.set_trace()
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
        
        self.sources = sources
        self.targets = targets
        
        # logging.warning("Tokenizing inputs... This may take some time...")
        # data_dict = preprocess(sources, targets, tokenizer)
        
        # self.input_ids = data_dict["input_ids"]
        # self.labels = data_dict["labels"]
    
    def __len__(self):
        return len(self.sources)
    
    def naive__getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
    
    def __getitem__(self, i):
        return dict(input_ids=self.sources[i], labels=self.targets[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    
    tokenizer: transformers.PreTrainedTokenizer
    
    def naive__call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        for instance in instances:
            source = instance['input_ids']
            target = instance['labels']
            sources.append(source)
            targets.append(target)
        
        data_dict = preprocess(sources, targets, self.tokenizer)
        input_ids, labels = data_dict['input_ids'], data_dict['labels']
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


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


def compute_loss(model, inputs, return_outputs=False):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.

    Subclass and override for custom behavior.
    """
    outputs = model(**inputs)
    if isinstance(outputs, dict) and "loss" not in outputs:
        raise ValueError(
            "The model did not return a loss from the inputs, only the following keys: "
            f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
        )
    # We don't use .loss here since the model may return tuples instead of ModelOutput.
    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    
    return (loss, outputs) if return_outputs else loss


def training_step(accelerator: Accelerator, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]],
                  training_args: TrainingArguments) -> torch.Tensor:
    model.train()
    with contextlib.nullcontext():
        loss = compute_loss(model, inputs)
    
    if torch.cuda.device_count() > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training
    
    accelerator.backward(loss)
    
    return loss.detach() / training_args.gradient_accumulation_steps


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
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
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
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(data_module['train_dataset'])
    train_loader = DataLoader(data_module["train_dataset"],
                              sampler=train_sampler,
                              batch_size=training_args.per_device_train_batch_size,
                              collate_fn=data_module["data_collator"])
    fsdp_config = {
        "use_orig_params": True,
        "device_id": rank,
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "auto_wrap_policy": partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=(LlamaDecoderLayer,)),
        "mixed_precision": MixedPrecision(param_dtype=torch.bfloat16),
    }
    model = FullyShardedDataParallel(model, **fsdp_config)
    if training_args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if training_args.torch_compile:
        model = torch.compile(model)
    model.train()
    optimizer = AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)
    num_epochs = int(training_args.num_train_epochs)
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    pbar = tqdm(range(num_training_steps))
    model.zero_grad()
    for epoch in range(num_epochs):
        for idx, batch0 in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch0.items() if k in ("input_ids", "attention_mask", "labels")}
            outputs = model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            model.zero_grad()
            max_memory = torch.cuda.max_memory_allocated()
            pbar.set_description(
                f"train_loss: {loss.item():.5f}, epoch: {epoch * idx / len(train_loader):.2f}, lr: {lr_scheduler.get_last_lr()[0]:.5f}, Memory: {max_memory / 1e9:.3f}G")
            pbar.update(1)
            idx += 1
            # if idx % training_args.eval_steps == 0:
            #     model.eval()
            #     eval(model, eval_loader, tokenizer)
            #     model.train()
    save_policy = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
    with FullyShardedDataParallel.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()
    if rank == 0:
        # print("torch save", training_args.output_dir)
        # torch.save(cpu_state, training_args.output_dir)
        print("save_pretrained", training_args.output_dir)
        model.save_pretrained(save_directory=training_args.output_dir, state_dict=cpu_state)
        tokenizer.save_pretrained(save_directory=training_args.output_dir)
    cleanup()


if __name__ == "__main__":
    torch.manual_seed(42)
    
    fsdp_main()
