# Lica — Text→SVG (retrieval baseline + seq2seq)

This repo contains:

1. **A tiny retrieval baseline** you can run instantly (no heavy deps) to verify end-to-end SVG output.  
2. **Three training scripts** for a T5-based Text→SVG model:  
   - `v1` — Minimal baseline  
   - `v2` — Laptop-safe training (AMP/BF16, grad accumulation, optional LoRA)  
   - `v3` — Grammar-aware targets + constrained decoding  
3. **Evaluators** for raster (SSIM, LPIPS, CLIPScore) and structure metrics (token_edit_norm).

---

## Quickstart: Retrieval Demo

Run a simple retrieval-based Text→SVG demo — no GPU required.

```bash
python scripts/retrieval_demo.py --query "blue square logo"
open outputs/demo_output.svg 
```
This confirms your environment and dependencies work before training.


## Environment Setup
```bash
# conda create -n lica-svgs python=3.10 -y && conda activate lica-svgs

pip install -r requirements.txt
```

## Windows tip:
If you encounter Unicode decode errors, enable UTF-8 mode:
```bash
setx PYTHONUTF8 1
```
## Data

We use the StarVector datasets
 on Hugging Face:

starvector/text2svg-stack — text-captioned SVGs (~2.18M)

starvector/svg-stack-simple — simplified SVGs (~1.29M)

### Prepare lightweight subsets for experimentation:
```bash
python scripts/prepare_data.py --dataset text2svg-stack --split train --sample 50000 --out data/train_50k.jsonl
python scripts/prepare_data.py --dataset text2svg-stack --split val   --sample 2000  --out data/val_2k.jsonl
```

Each JSONL row should look like:
```json
{"caption": "a red circle icon", "svg": "<svg ...>"}
```
## Train (three progressive scripts)
### Baseline — train_seq2seq_v1_baseline.py

Simple T5 seq2seq baseline with subword tokenization.
```bash
python scripts/train_seq2seq_v1_baseline.py \
  --train_jsonl data/train_50k.jsonl \
  --val_jsonl   data/val_2k.jsonl \
  --model_name  t5-small \
  --max_src_len 128 --max_tgt_len 512 \
  --batch_size 1 --epochs 2 \
  --output_dir outputs/t5_v1_baseline
```
### Laptop-Friendly — train_seq2seq_v2_lite_memory.py

Adds gradient accumulation, AMP/BF16 precision, and optional LoRA.
```bash
python scripts/train_seq2seq_v2_lite_memory.py \
  --train_jsonl data/train_50k.jsonl \
  --val_jsonl   data/val_2k.jsonl \
  --model_name  t5-small \
  --max_src_len 128 --max_tgt_len 640 \
  --batch_size 1 --epochs 2 \
  --grad_accum 4 \
  --use_lora \
  --output_dir outputs/t5_v2_lite_mem
```
### Grammar + Constrained Decoding — train_seq2seq_v3_grammar_constrained.py

Uses structured SVG tokenization and decoding rules for valid XML.

Requires:
```bash
scripts/svg_tokens.py

scripts/svg_constraints.py

python scripts/train_seq2seq_v3_grammar_constrained.py \
  --train_jsonl data/train_50k.jsonl \
  --val_jsonl   data/val_2k.jsonl \
  --model_name  t5-small \
  --max_src_len 128 --max_tgt_len 640 \
  --batch_size 1 --epochs 2 \
  --grad_accum 4 \
  --gen_len 640 \
  --output_dir outputs/t5_v3_grammar_constrained
```

This will also write:

outputs/t5_v3_grammar_constrained/predictions.jsonl
## Evaluation

Install rasterizers and metrics:
```bash
pip install skia-python cairosvg torch torchvision torchaudio torchmetrics lpips open-clip-torch pillow
```

Run the robust evaluator:
```bash
python scripts/evaluate_v3_full.py \
  --pred_jsonl outputs/<model_dir>/predictions.jsonl \
  --metrics raster_ssim,lpips,clipscore,token_edit \
  --size 224 \
  --debug
```

Example outputs:
```bash
Counts: {'lpips': 97, 'ssim': 97, 'clipscore': 100, 'token_edit_norm': 100, ...}
Aggregate: {'lpips': 0.225, 'ssim': 0.795, 'token_edit_norm': 1.06, ...}
Failures: {'raster': 0, 'ssim': 0, 'lpips': 0, 'clip': 0}
```
### PowerShell runners
# Baseline
```bash
python scripts/evaluate_v3_full.py --pred_jsonl outputs/t5_v1_baseline/predictions.jsonl --metrics raster_ssim,lpips,clipscore,token_edit --size 224 --debug

# Grammar Tokenizer
python scripts/evaluate_v3_full.py --pred_jsonl outputs/t5_v2_lite_mem/predictions.jsonl --metrics raster_ssim,lpips,clipscore,token_edit --size 224 --debug

# Grammar + Constrained
python scripts/evaluate_v3_full.py --pred_jsonl outputs/t5_v3_grammar_constrained/predictions.jsonl --metrics raster_ssim,lpips,clipscore,token_edit --size 224 --debug
```

## Repo Layout

    scripts/
      prepare_data.py
      retrieval_demo.py
      train_seq2seq_v1_baseline.py
      train_seq2seq_v2_lite_memory.py
      train_seq2seq_v3_grammar_constrained.py
      svg_tokens.py
      svg_constraints.py
      evaluate_v1_minimal.py
      evaluate_v2_raster.py
      evaluate_v3_full.py
      make_svg_comparison.py
    data/
    outputs/


## References

T5: Raffel et al., JMLR 2020

DeepSVG: Carlier et al., NeurIPS 2020

DiffVG: Li et al., SIGGRAPH 2020

CLIP: Radford et al., ICML 2021
