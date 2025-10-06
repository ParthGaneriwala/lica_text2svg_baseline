import argparse, json, os, re, math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from datasets import Dataset

import torch  # ⭐ needed for .generate and device handling

# Very simple tokenizer for SVG (replace with a grammar-aware tokenizer later)
def canonicalize_svg(svg: str) -> str:
    s = re.sub(r"\s+", " ", svg.strip())
    return s

def build_hf_dataset(train_jsonl, val_jsonl):
    def read_jsonl(p):
        # ⭐ Ensure Windows reads UTF-8 and never crashes on rare bytes
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            rows = [json.loads(l) for l in f]
        return Dataset.from_list(rows)
    return read_jsonl(train_jsonl), read_jsonl(val_jsonl)

@dataclass
class Example:
    caption: str
    svg: str

def preprocess(examples, tokenizer, max_src_len, max_tgt_len):
    captions = examples["caption"]
    svgs = [canonicalize_svg(s) for s in examples["svg"]]
    model_inputs = tokenizer(captions, max_length=max_src_len, truncation=True)
    enc = tokenizer(captions, max_length=max_src_len, truncation=True)
    lbl = tokenizer(text_target=svgs, max_length=max_tgt_len, truncation=True)
    model_inputs = {**enc, "labels": lbl["input_ids"]}
    return model_inputs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--val_jsonl", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="t5-small")
    ap.add_argument("--max_src_len", type=int, default=128)
    ap.add_argument("--max_tgt_len", type=int, default=512)
    ap.add_argument("--output_dir", type=str, default="outputs/t5_text2svg")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    train_ds, val_ds = build_hf_dataset(args.train_jsonl, args.val_jsonl)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    fn = lambda batch: preprocess(batch, tokenizer, args.max_src_len, args.max_tgt_len)
    train_ds = train_ds.map(fn, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(fn, batched=True, remove_columns=val_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # ⭐ Cleaner device/bfloat handling
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="steps",
        eval_steps=2000,
        save_steps=2000,
        logging_steps=200,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=use_bf16,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    trainer.train()
    os.makedirs(args.output_dir, exist_ok=True)

    # quick qualitative dump on val set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ⭐
    model.to(device)  # ⭐
    val_raw = [json.loads(l) for l in open(args.val_jsonl, "r", encoding="utf-8", errors="replace")]  # ⭐
    preds = []
    for r in val_raw[:100]:
        inputs = tokenizer([r["caption"]], return_tensors="pt", truncation=True, max_length=args.max_src_len).to(device)  # ⭐
        with torch.no_grad():
            out = model.generate(**inputs, max_length=args.max_tgt_len, num_beams=4)
        svg_pred = tokenizer.decode(out[0], skip_special_tokens=True)
        preds.append({"caption": r["caption"], "svg_pred": svg_pred, "svg_gt": r["svg"]})

    with open(Path(args.output_dir) / "predictions.jsonl", "w", encoding="utf-8") as f:  # ⭐
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Wrote qualitative predictions to {Path(args.output_dir) / 'predictions.jsonl'}")

if __name__ == "__main__":
    main()
