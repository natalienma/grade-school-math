import os
import re
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
import torch.utils.data.distributed
import transformers
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import get_scheduler
from transformers.trainer_pt_utils import LabelSmoother

from dataset import read_jsonl

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


class GSMDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, examples, loss_on_prefix=False, padding_left=True, debug=False):
        self.tokenizer = tokenizer
        self.examples = examples
        self.loss_on_prefix = loss_on_prefix
        self.padding_left = padding_left
        self.debug = debug
        self.sep_token = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        self.processed_examples = [
            self.process_example(example, loss_on_prefix, padding_left) for example in examples
        ]
    
    def process_example(self, example, loss_on_prefix, padding_left):
        q_ids = self.tokenizer.encode(example["question"], add_special_tokens=False)
        a_ids = self.tokenizer.encode(example["answer"], add_special_tokens=False)
        input_ids = q_ids + a_ids
        max_len = 512
        max_q_len = 512
        input_padding_length = max(0, max_len - len(input_ids))
        q_padding_length = max(0, max_q_len - len(q_ids))
        q_length = len(q_ids)
        if padding_left:
            input_ids = [self.tokenizer.pad_token_id] * input_padding_length + input_ids
            q_ids = [self.tokenizer.pad_token_id] * q_padding_length + q_ids
        else:
            input_ids += [self.tokenizer.pad_token_id] * input_padding_length
            q_ids += [self.tokenizer.pad_token_id] * q_padding_length
        label_ids = input_ids[:]
        if not loss_on_prefix:
            label_ids[:q_length] = [IGNORE_TOKEN_ID] * q_length
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
        if self.debug:
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


def main():
    global autocast
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    print(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    train_examples = get_examples(data_args.data_path)
    train_dset = GSMDataset(tokenizer, train_examples, padding_left=data_args.padding_left, loss_on_prefix=data_args.loss_on_prefix, debug=data_args.data_debug)
    train_loader = DataLoader(train_dset, batch_size=training_args.per_device_train_batch_size, shuffle=True)
    eval_examples = get_examples(data_args.eval_data_path)[:500]
    eval_dset = GSMDataset(tokenizer, eval_examples, padding_left=data_args.eval_padding_left, loss_on_prefix=data_args.loss_on_prefix)
    eval_loader = DataLoader(eval_dset, batch_size=training_args.per_device_eval_batch_size, shuffle=False)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        n_gpus = 1
    
    print("n_gpus: ", n_gpus)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    model = model.to(device)
    # if n_gpus > 1:
    #     model = torch.nn.DataParallel(model)

    print("training_args.tf32", training_args.tf32)
    print("torch.backends.cuda.matmul.allow_tf32", torch.backends.cuda.matmul.allow_tf32)
    print("torch.backends.cudnn.allow_tf32", torch.backends.cudnn.allow_tf32)
    if training_args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if training_args.bf16:
        from torch.cuda.amp import autocast
    if training_args.torch_compile:
        model = torch.compile(model)
    # model = DDP(model, device_ids=[gpu_id])
    model.train()
    opt = AdamW(model.parameters(), lr=training_args.learning_rate)
    
    num_epochs = int(training_args.num_train_epochs)
    num_training_steps = num_epochs * len(train_loader)
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=opt,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    pbar = tqdm(range(num_training_steps))
    idx = 0
    for epoch in range(num_epochs):
        for batch0 in train_loader:
            batch = {k: v.to(device) for k, v in batch0.items() if k in ("input_ids", "attention_mask")}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(**batch, labels=batch0["labels"])
                loss = outputs[0]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            lr_scheduler.step()
            opt.zero_grad()
            pbar.update(1)
            pbar.set_description(f"train_loss: {loss.item():.5f}, epoch: {epoch}, lr: {lr_scheduler.get_last_lr()[0]:.5f}")
            idx += 1
            if idx % training_args.eval_steps == 0:
                model.eval()
                eval(model, eval_loader, tokenizer)
                model.train()
    
        model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
