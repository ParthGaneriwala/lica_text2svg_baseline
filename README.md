# Lica — Text→SVG Baseline (retrieval + seq2seq skeleton)

This repo contains:
1) **A tiny, fully runnable retrieval baseline** over a toy sample (no external deps), so you can verify end-to-end SVG outputs immediately.
2) **Training skeletons** for a stronger seq2seq model on the Hugging Face `starvector/text2svg-stack` dataset (requires installing deps).

> If you only want to run something *now*, use the retrieval demo:
```bash
python scripts/retrieval_demo.py --query "blue square logo"
open outputs/demo_output.svg  # or view in any browser
```

## Environment (for full training)
```bash
# optional: conda create -n lica-svgs python=3.10 -y && conda activate lica-svgs
pip install -r requirements.txt
```

## Data (full run)
We rely on the StarVector datasets published on Hugging Face:
- `starvector/text2svg-stack` — text-captioned SVGs (≈2.18M rows)
- `starvector/svg-stack-simple` — simplified SVGs (≈1.29M rows)

Prepare a small subset for quick iteration:
```bash
python scripts/prepare_data.py --dataset text2svg-stack --split train --sample 50000 --out data/train_50k.jsonl
python scripts/prepare_data.py --dataset text2svg-stack --split val   --sample 2000  --out data/val_2k.jsonl
```

## Train (seq2seq skeleton)
```bash
python scripts/train_seq2seq.py   --train_jsonl data/train_50k.jsonl   --val_jsonl   data/val_2k.jsonl   --model_name  t5-small   --max_src_len 128 --max_tgt_len 512   --output_dir outputs/t5_text2svg
```

## Evaluate
```bash
python scripts/evaluate.py   --pred_jsonl outputs/t5_text2svg/predictions.jsonl   --metrics raster_ssim,lpips,clipscore,token_edit
```

## Notes
- The retrieval baseline here is intentionally simple and dependency-free.
- The seq2seq code is a *skeleton* meant to be expanded. It uses a grammar-aware tokenizer stub you can replace with your preferred SVG tokenizer.
- Optional differentiable refinement (DiffVG / pydiffvg) can be added later.
