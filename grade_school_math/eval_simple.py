import copy
import json
import logging
import os
import re
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Dict, Sequence

import torch
import torch.distributed as dist
import torch.utils.data.distributed
import transformers
from datasets import Dataset
from torch.distributed.fsdp import FullyShardedDataParallel, StateDictType, ShardingStrategy
from torch.distributed.fsdp.wrap import _module_wrap_policy
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoConfig
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.trainer_pt_utils import LabelSmoother
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset import read_jsonl
from train_fsdp_alpaca import SimpleDataset, collate_fn, collate_fn_eval

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
        # do_sample=True,
        # num_beams=5,
        max_new_tokens=256,
        # top_k=50,
        # top_p=0.95,
        # num_return_sequences=5,
        pad_token_id=tokenizer.eos_token_id
    )
    print(f"Number of batches: {len(eval_dataloader)}")
    pred_ans_list = []
    gold_ans_list = []
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            try:
                batch_output = model.module.generate(
                    input_ids=batch['input_ids'].to(model.device),
                    attention_mask=batch["attention_mask"].to(model.device),
                    generation_config=generation_config,
                    return_dict_in_generate=True
                )
            except RuntimeError as e:
                print(e)
                raise e
        outputs_string = tokenizer.batch_decode(batch_output.sequences, skip_special_tokens=True)
        for gold_ans1, pred_ans in zip(batch['examples'], outputs_string):
            gold_ans = gold_ans1["answer"]
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
    
    correct_tensor = torch.tensor(cor, device=model.device)
    invalid_tensor = torch.tensor(invalid, device=model.device)
    total_tensor = torch.tensor(len(rg), device=model.device)
    
    torch.distributed.all_reduce(correct_tensor, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(invalid_tensor, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)
    
    if torch.distributed.get_rank() == 0:
        print(f"Rank: 0\nCorrect: {correct_tensor.item()}, "
              f"Total: {total_tensor.item()}, "
              f"Accuracy: {correct_tensor.item() / total_tensor.item():.3f}, "
              f"Invalid: {invalid_tensor.item()}")


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
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config
    )
    tokenizer = AutoTokenizer.from_pretrained(
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
    tokenizer.add_special_tokens({
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN})
    eval_dset = SimpleDataset(data_path=data_args.eval_data_path)
    eval_sampler = DistributedSampler(eval_dset, num_replicas=world_size, rank=rank)
    eval_loader = DataLoader(eval_dset, batch_size=training_args.per_device_eval_batch_size,
                             collate_fn=lambda x: collate_fn_eval(x, tokenizer),
                             sampler=eval_sampler,
                             shuffle=False)
    tokenizer.truncation_side = 'left'
    torch.cuda.set_device(rank)
    model.to(torch.cuda.current_device())
    model = DDP(model, device_ids=[torch.cuda.current_device()])
    model.eval()
    
    if training_args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if training_args.torch_compile:
        model = torch.compile(model)
    eval(model, eval_loader, tokenizer)
    cleanup()


if __name__ == "__main__":
    torch.manual_seed(42)
    
    fsdp_main()
