import argparse, json, os, re
from pathlib import Path
from dataclasses import dataclass
import inspect
from svg_tokens import tokenize_svg, detokenize, special_tokens
from svg_constraints import SVGConstrainedProcessor
import inspect

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ---- tiny svg canonicalizer ----
def canonicalize_svg(svg: str) -> str:
    return re.sub(r"\s+", " ", svg.strip())

def build_hf_dataset(train_jsonl, val_jsonl):
    def read_jsonl(p):
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            rows = [json.loads(l) for l in f]
        return Dataset.from_list(rows)
    return read_jsonl(train_jsonl), read_jsonl(val_jsonl)

@dataclass
class Example:
    caption: str
    svg: str

def preprocess(examples, tokenizer, max_src_len, max_tgt_len, use_svg_grammar=False):
    captions = examples["caption"]
    if use_svg_grammar:
        # tokens → space-joined string (deterministic)
        svgs_tok = [" ".join(tokenize_svg(s)) for s in examples["svg"]]
    else:
        svgs_tok = [re.sub(r"\s+", " ", s.strip()) for s in examples["svg"]]
    enc = tokenizer(captions, max_length=max_src_len, truncation=True)
    lbl = tokenizer(text_target=svgs_tok, max_length=max_tgt_len, truncation=True)
    enc["labels"] = lbl["input_ids"]
    return enc


def make_training_args(args, use_bf16: bool):
    # Build kwargs compatible with the installed transformers version
    sig = inspect.signature(TrainingArguments.__init__)
    supported = set(sig.parameters.keys())

    kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "save_total_limit": 2,
        "report_to": "none",
    }

    # Optional, if present in your version:
    if "logging_steps" in supported:
        kwargs["logging_steps"] = 200
    if "save_steps" in supported:
        kwargs["save_steps"] = 2000
    if "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = "steps"
        if "eval_steps" in supported:
            kwargs["eval_steps"] = 2000
    elif "evaluate_during_training" in supported:
        # very old transformers (<3.x)
        kwargs["evaluate_during_training"] = True

    if "bf16" in supported:
        kwargs["bf16"] = use_bf16

    # Nice-to-haves if available:
    if "remove_unused_columns" in supported:
        kwargs["remove_unused_columns"] = True
    if "load_best_model_at_end" in supported:
        kwargs["load_best_model_at_end"] = False

    return TrainingArguments(**kwargs)

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
    ap.add_argument("--use_svg_grammar", action="store_true")
    ap.add_argument("--gen_len", type=int, default=640)
    args = ap.parse_args()

    train_ds, val_ds = build_hf_dataset(args.train_jsonl, args.val_jsonl)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    new_toks = special_tokens()
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": new_toks})
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    fn = lambda batch: preprocess(batch, tokenizer, args.max_src_len, args.max_tgt_len, args.use_svg_grammar)
    train_ds = train_ds.map(fn, batched=True, remove_columns=train_ds.column_names)
    val_ds   = val_ds.map(fn, batched=True, remove_columns=val_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = make_training_args(args, use_bf16)

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

    from transformers import LogitsProcessorList
    id2tok = {tokenizer.convert_tokens_to_ids(t): t for t in new_toks if tokenizer.convert_tokens_to_ids(t) != tokenizer.unk_token_id}
    tok2id = {v:k for k,v in id2tok.items()}
    proc = SVGConstrainedProcessor(id2tok=id2tok, tok2id=tok2id)
    lp = LogitsProcessorList([proc])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    val_raw = [json.loads(l) for l in open(args.val_jsonl, "r", encoding="utf-8", errors="replace")]
    preds = []
    for r in val_raw[:100]:
        inputs = tokenizer([r["caption"]], return_tensors="pt", truncation=True, max_length=args.max_src_len).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_length=args.gen_len, num_beams=3, length_penalty=1.1,
                logits_processor=lp, early_stopping=False
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        # If grammar used, text is tokens separated by spaces then detokenize to SVG
        if args.use_svg_grammar:
            toks = text.split()
            svg_pred = detokenize(toks)
        else:
            svg_pred = text
        preds.append({"caption": r["caption"], "svg_pred": svg_pred, "svg_gt": r["svg"]})

    with open(Path(args.output_dir) / "predictions.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote qualitative predictions to {Path(args.output_dir) / 'predictions.jsonl'}")

if __name__ == "__main__":
    main()
