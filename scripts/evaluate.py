import argparse, json, io, os, math, tempfile, subprocess, warnings, re
from pathlib import Path
from PIL import Image
import numpy as np

# Optional imports guarded by try-except
try:
    import lpips
except Exception as e:
    lpips = None
try:
    import torch
except Exception as e:
    torch = None
try:
    from torchmetrics.functional.image.ssim import structural_similarity_index_measure as ssim_fn
except Exception as e:
    ssim_fn = None

def rasterize_svg(svg_str, size=128):
    # Prefer cairosvg if available; else fallback to pillow via temporary PNG (requires cairosvg)
    try:
        import cairosvg, io
        png_bytes = cairosvg.svg2png(bytestring=svg_str.encode("utf-8"), output_width=size, output_height=size)
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        return img
    except Exception as e:
        raise RuntimeError("Rasterization requires cairosvg: pip install cairosvg") from e

def compute_lpips(x, y):
    if lpips is None or torch is None:
        return None
    loss_fn = lpips.LPIPS(net='alex')
    x_t = torch.from_numpy(np.array(x).transpose(2,0,1)).float().unsqueeze(0) / 255.0
    y_t = torch.from_numpy(np.array(y).transpose(2,0,1)).float().unsqueeze(0) / 255.0
    return float(loss_fn(x_t, y_t).item())

def compute_ssim(x, y):
    if ssim_fn is None or torch is None:
        return None
    x_t = torch.from_numpy(np.array(x).transpose(2,0,1)).float().unsqueeze(0) / 255.0
    y_t = torch.from_numpy(np.array(y).transpose(2,0,1)).float().unsqueeze(0) / 255.0
    return float(ssim_fn(x_t, y_t, data_range=1.0))

def token_edit_distance(a, b):
    a_toks = re.findall(r"[A-Za-z]+|[-+]?\d*\.?\d+|[#][0-9A-Fa-f]+|[<>/='\";:(),]", a)
    b_toks = re.findall(r"[A-Za-z]+|[-+]?\d*\.?\d+|[#][0-9A-Fa-f]+|[<>/='\";:(),]", b)
    # Levenshtein distance
    dp = [[0]*(len(b_toks)+1) for _ in range(len(a_toks)+1)]
    for i in range(len(a_toks)+1): dp[i][0] = i
    for j in range(len(b_toks)+1): dp[0][j] = j
    for i in range(1,len(a_toks)+1):
        for j in range(1,len(b_toks)+1):
            cost = 0 if a_toks[i-1]==b_toks[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[-1][-1], len(a_toks), len(b_toks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", type=str, required=True, help="predictions.jsonl containing svg_pred and svg_gt")
    ap.add_argument("--metrics", type=str, default="raster_ssim,lpips,token_edit")
    args = ap.parse_args()
    metrics = set(m.strip() for m in args.metrics.split(","))

    preds = [json.loads(l) for l in open(args.pred_jsonl)]
    rows = []
    for p in preds:
        row = {}
        if "raster_ssim" in metrics or "lpips" in metrics:
            try:
                img_pred = rasterize_svg(p["svg_pred"])
                img_gt   = rasterize_svg(p["svg_gt"])
            except Exception as e:
                img_pred = img_gt = None

        if "raster_ssim" in metrics and img_pred is not None and img_gt is not None:
            row["ssim"] = compute_ssim(img_pred, img_gt)

        if "lpips" in metrics and img_pred is not None and img_gt is not None:
            row["lpips"] = compute_lpips(img_pred, img_gt)

        if "token_edit" in metrics:
            dist, na, nb = token_edit_distance(p["svg_pred"], p["svg_gt"])
            row["token_edit"] = dist
            row["len_pred"] = na
            row["len_gt"] = nb

        rows.append(row)

    # simple mean report
    agg = {}
    for k in rows[0].keys():
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            agg[k] = float(np.mean(vals))
    print("Aggregate:", agg)

if __name__ == "__main__":
    main()
