#!/usr/bin/env python
import argparse, json, os, re
from pathlib import Path
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments,
    LogitsProcessorList
)

# local modules (ensure these files exist in scripts/)
from svg_tokens import tokenize_svg, detokenize, special_tokens
from svg_constraints import SVGConstrainedProcessor

def read_jsonl(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [json.loads(l) for l in f]

def build_hf_dataset(train_jsonl, val_jsonl):
    return Dataset.from_list(read_jsonl(train_jsonl)), Dataset.from_list(read_jsonl(val_jsonl))

def preprocess_fn(max_src_len, max_tgt_len, tokenizer, use_svg_grammar):
    def _fn(batch):
        caps = batch["caption"]
        if use_svg_grammar:
            # tokens as a space-joined string to preserve discrete symbols
            svgs = [" ".join(tokenize_svg(s)) for s in batch["svg"]]
        else:
            svgs = [re.sub(r"\s+", " ", s.strip()) for s in batch["svg"]]
        x = tokenizer(caps, max_length=max_src_len, truncation=True)
        y = tokenizer(text_target=svgs, max_length=max_tgt_len, truncation=True)
        x["labels"] = y["input_ids"]
        return x
    return _fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--model_name", default="t5-small")
    ap.add_argument("--max_src_len", type=int, default=128)
    ap.add_argument("--max_tgt_len", type=int, default=640)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--gen_len", type=int, default=640)
    ap.add_argument("--use_svg_grammar", action="store_true", default=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    # Fast matmuls on Ampere
    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

    train_ds, val_ds = build_hf_dataset(args.train_jsonl, args.val_jsonl)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # add grammar special tokens so tokenizer doesn’t split them
    added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens()})
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))

    # gradient checkpointing to save VRAM
    try: model.gradient_checkpointing_enable()
    except Exception: pass

    # map datasets
    fn = preprocess_fn(args.max_src_len, args.max_tgt_len, tokenizer, args.use_svg_grammar)
    train_ds = train_ds.map(fn, batched=True, remove_columns=train_ds.column_names)
    val_ds   = val_ds.map(fn, batched=True, remove_columns=val_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    os.makedirs(args.output_dir, exist_ok=True)

    bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        predict_with_generate=False,
        remove_unused_columns=True,
        dataloader_num_workers=2,
        fp16=not bf16_ok,
        bf16=bf16_ok,
        report_to=[],
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tokenizer, data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ---- Qualitative predictions with constrained decoding ----
    print("Generating qualitative predictions with constraints…")
    # id↔token maps only for our special tokens
    spec = special_tokens()
    id2tok = {tokenizer.convert_tokens_to_ids(t): t for t in spec
              if tokenizer.convert_tokens_to_ids(t) != tokenizer.unk_token_id}
    tok2id = {v: k for k, v in id2tok.items()}
    proc = SVGConstrainedProcessor(id2tok=id2tok, tok2id=tok2id, min_struct_tokens=12)
    lp = LogitsProcessorList([proc])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # load raw validation rows (not tokenized) to get captions/svg
    val_raw = read_jsonl(args.val_jsonl)
    preds = []
    for r in val_raw[:100]:  # small sample for quick inspection
        inputs = tokenizer([r["caption"]], return_tensors="pt",
                           truncation=True, max_length=args.max_src_len).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_length=args.gen_len,
                min_length=min(args.gen_len // 3, 256),
                num_beams=3, length_penalty=1.15,
                logits_processor=lp, early_stopping=False
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        if args.use_svg_grammar:
            svg_pred = detokenize(text.split())
        else:
            svg_pred = text
        preds.append({"caption": r["caption"], "svg_pred": svg_pred, "svg_gt": r["svg"]})

    outp = Path(args.output_dir) / "predictions.jsonl"
    with open(outp, "w", encoding="utf-8") as f:
        for p in preds: f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("Wrote", outp)

if __name__ == "__main__":
    main()
