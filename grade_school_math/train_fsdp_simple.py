import contextlib
import copy
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Dict, Sequence, Union, Any, List, Tuple

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
        default="data/train.jsonl", metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default="data/test.jsonl", metadata={"help": "Path to the evaluation data."}
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
            print("batch: ", batch['input_ids'], batch['attention_mask'])
            batch_output = model.generate(
                input_ids=batch['input_ids'].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                generation_config=generation_config,
                return_dict_in_generate=True
            )
        outputs_string = tokenizer.batch_decode(batch_output.sequences, skip_special_tokens=True)
        for gold_ans, pred_ans in zip(batch['examples']["answer"], outputs_string):
            gold_ext = extract_answer(gold_ans)
            pred_ext = extract_answer(pred_ans)
            gold_ans_list.append(gold_ext)
            pred_ans_list.append(pred_ext)
            # print("GOLD: ", gold_ext, gold_ans)
            # print("PRED: ", pred_ext, pred_ans)
    
    cor = 0
    invalid = 0
    rg = range(min(len(pred_ans_list), len(gold_ans_list)))
    for i in rg:
        if pred_ans_list[i] != INVALID_ANS and abs(float(pred_ans_list[i]) - float(gold_ans_list[i])) < 1e-4:
            cor += 1
        if pred_ans_list[i] == INVALID_ANS:
            invalid += 1
    print("correct: %d, total: %d, acc: %.3f, invalid: %d" % (cor, len(list(rg)), cor / len(list(rg)), invalid))


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


class SimpleDataset(Dataset):
    def __init__(self, data_path: str):
        with open(data_path, 'r') as f:
            lines = f.readlines()
        self.examples = [json.loads(line.strip()) for line in lines]
        if data_path.startswith("data/train"):
            sample_size = len(self.examples) // 20
        else:
            sample_size = len(self.examples) // 5
        print("data path:", data_path, "sample size:", sample_size)
        import random
        random.seed(42)
        self.examples = random.sample(self.examples, sample_size)
        self.inputs = [ex["question"] for ex in self.examples]
        self.labels = [ex["answer"] for ex in self.examples]
        self.inputs_labels = [f"{x}{y}" for x, y in zip(self.inputs, self.labels)]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx) -> Tuple[str, str, str, str]:
        return self.inputs[idx], self.labels[idx], self.inputs_labels[idx], self.examples[idx]


def collate_fn(batch: List[Tuple[str, str]], tokenizer) -> dict:
    inputs, labels, inputs_labels, _ = zip(*batch)
    inputs_tok = tokenizer(inputs, padding="longest", truncation=True, max_length=tokenizer.model_max_length,
                           return_tensors="pt")
    inputs_labels_tok = tokenizer(inputs_labels, padding="longest", truncation=True,
                                  max_length=tokenizer.model_max_length,
                                  return_tensors="pt")
    input_ids_lens = [item.ne(tokenizer.pad_token_id).sum().item() for item in inputs_tok['input_ids']]
    label_inputs = inputs_labels_tok['input_ids'].clone()
    for i, input_ids_len in enumerate(input_ids_lens):
        label_inputs[i, :input_ids_len] = tokenizer.pad_token_id
    label_inputs[label_inputs == tokenizer.pad_token_id] = IGNORE_INDEX
    return {"input_ids": inputs_labels_tok['input_ids'], "attention_mask": inputs_labels_tok['attention_mask'],
            "labels": label_inputs}


def collate_fn_eval(batch: List[Tuple[str, str]], eval_tokenizer) -> dict:
    inputs, _, _, examples = zip(*batch)
    inputs_tok = eval_tokenizer(inputs,
                                padding="longest",
                                truncation=True,
                                max_length=eval_tokenizer.model_max_length,
                                return_tensors="pt")
    return {"input_ids": inputs_tok['input_ids'], "attention_mask": inputs_tok['attention_mask'],
            "examples": examples}


def fsdp_main():
    global batch0
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print("rank: ", rank, "world_size: ", world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
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
    eval_tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
        eval_tokenizer.add_special_tokens({'pad_token': DEFAULT_PAD_TOKEN})
    tokenizer.add_special_tokens({
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN
    })
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
    train_dset = SimpleDataset(data_path=data_args.data_path)
    train_loader = DataLoader(train_dset, batch_size=training_args.per_device_train_batch_size,
                              collate_fn=lambda x: collate_fn(x, tokenizer))
    eval_dset = SimpleDataset(data_path=data_args.eval_data_path)
    eval_loader = DataLoader(eval_dset, batch_size=training_args.per_device_eval_batch_size,
                             collate_fn=lambda x: collate_fn_eval(x, eval_tokenizer))
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    pbar = tqdm(range(num_training_steps))
    model.zero_grad()
    total_batches = len(train_loader)
    total_samples = len(train_dset)
    epoch_time_start = time.time()
    batch_size = training_args.per_device_train_batch_size
    for epoch in range(num_epochs):
        batch_time_start = time.time()
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
            batch_time_end = time.time()
            batch_duration = batch_time_end - batch_time_start
            batch_time_start = time.time()

            samples_per_second = batch_size / batch_duration
            if rank == 0:
                print(f"Epoch: {epoch + 1}/{num_epochs}, Batch: {idx + 1}/{total_batches}, Loss: {loss.item():.4f}, Samples/sec: {samples_per_second:.2f}")
        epoch_duration = time.time() - epoch_time_start
        epoch_time_start = time.time()
        if rank == 0:
            print(f"Completed Epoch {epoch + 1}/{num_epochs}, Duration: {epoch_duration:.2f} seconds, Avg samples/sec: {total_samples / epoch_duration:.2f}")
        # if idx % training_args.eval_steps == 0:
            #     model.eval()
            #     eval(model, eval_loader, eval_tokenizer)
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
