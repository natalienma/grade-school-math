# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import json
import math
import os
import pathlib
import re
import time
from dataclasses import dataclass, field
from typing import Optional
import torch._dynamo

# torch._dynamo.config.suppress_errors = True
print("torch._dynamo.config.automatic_dynamic_shapes", torch._dynamo.config.automatic_dynamic_shapes)
torch._dynamo.config.automatic_dynamic_shapes = False
print("torch._dynamo.config.automatic_dynamic_shapes", torch._dynamo.config.automatic_dynamic_shapes)
import torch
import transformers
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Trainer, GenerationConfig, AutoConfig
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_examples(path):
    path = os.path.join("data/", f"{path}.jsonl")
    examples = read_jsonl(path)
    
    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + " " + "<|endoftext|>")
    
    print(f"{path} got {len(examples)} examples")
    return examples


class GSMDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, examples, loss_on_prefix=False):
        self.tokenizer = tokenizer
        self.examples = examples
        self.loss_on_prefix = loss_on_prefix
        # self.sep_token = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        
        self.processed_examples = [
            self.process_example(example) for example in examples
        ]
    
    def process_example(self, example):
        q_ids = self.tokenizer.encode(example["question"],
                                      padding="max_length",
                                      max_length=512,
                                      truncation=True)
        qa_ids = self.tokenizer.encode(example["question"] + example["answer"],
                                       padding="max_length",
                                       max_length=512,
                                       truncation=True)
        input_ids = qa_ids
        label_ids = qa_ids[:]
        attention_mask = torch.tensor(input_ids, dtype=torch.long).ne(self.tokenizer.pad_token_id)
        q_attention_mask = torch.tensor(q_ids, dtype=torch.long).ne(self.tokenizer.pad_token_id)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": attention_mask,
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "q_ids": torch.tensor(q_ids, dtype=torch.long),
            "q_attention_mask": q_attention_mask
        }
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        print({
            "input_ids": self.processed_examples[idx]["input_ids"],
            "attention_mask": self.processed_examples[idx]["attention_mask"],
            "labels": self.processed_examples[idx]["labels"],
            "q_ids": self.processed_examples[idx]["q_ids"],
            "q_attention_mask": self.processed_examples[idx]["q_attention_mask"],
            "examples": self.examples[idx]
        })
        return {
            "input_ids": self.processed_examples[idx]["input_ids"],
            "attention_mask": self.processed_examples[idx]["attention_mask"],
            "labels": self.processed_examples[idx]["labels"],
            "q_ids": self.processed_examples[idx]["q_ids"],
            "q_attention_mask": self.processed_examples[idx]["q_attention_mask"],
            "examples": self.examples[idx]
        }
    
    @property
    def max_len(self):
        return max(len(item["input_ids"]) for item in self.processed_examples)
    
    @property
    def max_q_len(self):
        return max(len(item["q_ids"]) for item in self.processed_examples)


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


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False
    loss_on_prefix: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
    
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
            trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


from transformers import TrainerCallback


class EvaluationAccuracyCallback(TrainerCallback):
    def __init__(self, model, tokenizer, eval_dataloader, generation_config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.eval_dataloader = eval_dataloader
    
    def on_evaluate(self, args, state, control, **kwargs):
        pred_ans_list = []
        gold_ans_list = []
        for batch in tqdm(self.eval_dataloader):
            with torch.no_grad():
                batch_output = self.model.generate(
                    input_ids=batch['q_ids'].to(self.model.device),
                    attention_mask=batch["q_attention_mask"].to(self.model.device),
                    generation_config=self.generation_config,
                    return_dict_in_generate=True
                )
            outputs_string = self.tokenizer.batch_decode(batch_output.sequences, skip_special_tokens=True)
            for gold_ans, pred_ans in zip(batch['examples']["answer"], outputs_string):
                gold_ext = extract_answer(gold_ans)
                pred_ext = extract_answer(pred_ans)
                gold_ans_list.append(gold_ext)
                pred_ans_list.append(pred_ext)
                print("GOLD: ", gold_ans)
                print("PRED: ", pred_ans)
        
        cor = 0
        invalid = 0
        rg = range(min(len(pred_ans_list), len(gold_ans_list)))
        for i in rg:
            if pred_ans_list[i] != INVALID_ANS and abs(float(pred_ans_list[i]) - float(gold_ans_list[i])) < 1e-4:
                cor += 1
            if pred_ans_list[i] == INVALID_ANS:
                invalid += 1
        print("correct: %d, total: %d, acc: %.3f, invalid: %d" % (cor, len(list(rg)), cor / len(list(rg)), invalid))


def evaluate_on_gpu(model, tokenizer, batch, generation_config):
    with torch.no_grad():
        batch_output = model.generate(
            input_ids=batch['q_ids'].cuda(),
            attention_mask=batch["q_attention_mask"].cuda(),
            generation_config=generation_config,
            return_dict_in_generate=True
        )
    outputs_string = tokenizer.batch_decode(batch_output.sequences, skip_special_tokens=True)
    gold_ans_list = []
    pred_ans_list = []
    for gold_ans, pred_ans in zip(batch['examples']["answer"], outputs_string):
        gold_ext = extract_answer(gold_ans)
        pred_ext = extract_answer(pred_ans)
        gold_ans_list.append(gold_ext)
        pred_ans_list.append(pred_ext)
    return gold_ans_list, pred_ans_list


def train():
    global local_rank
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False
    
    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     config=config,
    #     cache_dir=training_args.cache_dir,
    # )
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    model = transformers.AutoModelForCausalLM.from_config(config)
    # model.init_weights()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    eval_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    eval_tokenizer.pad_token = eval_tokenizer.unk_token
    start_time = time.time()
    train_examples = get_examples(data_args.data_path)
    train_dset = GSMDataset(tokenizer, train_examples, loss_on_prefix=data_args.loss_on_prefix)
    # eval_examples = get_examples("test.jsonl")
    eval_examples = get_examples(data_args.eval_data_path)[:500]
    eval_dset = GSMDataset(eval_tokenizer, eval_examples, loss_on_prefix=data_args.loss_on_prefix)
    eval_dataloader = DataLoader(eval_dset, batch_size=training_args.per_device_eval_batch_size, shuffle=False,
                                 num_workers=4)
    end_time = time.time()
    print(f"Data loading took {(end_time - start_time):.2f} seconds.")
    data_module = dict(train_dataset=train_dset, eval_dataset=eval_dset)
    generation_config = GenerationConfig(
        # temperature=0.7,
        # do_sample=False,
        # num_beams=1,
        max_new_tokens=256,
        # num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
        callbacks=[EvaluationAccuracyCallback(model, eval_tokenizer, eval_dataloader, generation_config)]
    )
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    # Save model
    model.config.use_cache = True
    trainer.save_state()
    # trainer_save_model_safe(trainer)
    trainer.save_model()


if __name__ == "__main__":
    train()
