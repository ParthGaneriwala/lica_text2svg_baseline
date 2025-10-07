#!/usr/bin/env python
import argparse, io, json, os, re, numpy as np
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET

# backends
try: import skia
except Exception: skia = None
try: import cairosvg
except Exception: cairosvg = None

# metrics
try: import torch
except Exception: torch = None
try:
    from torchmetrics.functional.image.ssim import structural_similarity_index_measure as ssim_fn
except Exception:
    ssim_fn = None
try: import lpips
except Exception: lpips = None
try: import open_clip
except Exception: open_clip = None

# ---------- SVG auto-fix ----------
_ALLOWED_EMPTY = {"path","rect","circle","ellipse","line","polyline","polygon","stop"}
_SVG_ROOT_TMPL = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256">{content}</svg>'

def _strip_code_fences(s):
    s = s.strip()
    s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s)
    s = s.replace("<pad>","").replace("</s>","").replace("<s>","")
    s = re.sub(r"<extra_id_\d+>", "", s)
    return s.strip()

def _quote_attrs(s):  # foo=bar -> foo="bar"
    return re.sub(r'(\s[\w:-]+)=([#\w\.\-+,/]+)(?=[\s>/])', r'\1="\2"', s)

def _self_close(s):
    def repl(m):
        tag, attrs = m.group(1).lower(), m.group(2)
        if tag in _ALLOWED_EMPTY:
            return f"<{tag}{attrs}/>"
        return m.group(0)
    return re.sub(r"<(\w+)([^>/]*?)>(?!\s*</\1>)", repl, s)

def _sanitize(s):
    return re.sub(r"</?(script|foreignObject|iframe)[^>]*>", "", s, flags=re.I)

def _wrap_root(s):
    if "<svg" in s.lower(): return s
    frag = s[s.find("<"): s.rfind(">")+1] if "<" in s and ">" in s else s
    return _SVG_ROOT_TMPL.format(content=frag)

def _escape_amp(s): return re.sub(r"&(?![a-zA-Z#0-9]+;)", "&amp;", s)

def svg_autofix(raw: str):
    if not raw or not isinstance(raw, str): return None
    s = _strip_code_fences(raw)
    m = re.search(r"<(svg|path|rect|circle|polygon|polyline|line|g|defs|linearGradient|radialGradient)\b.*", s, flags=re.I|re.S)
    if m: s = s[m.start():]
    s = _sanitize(_quote_attrs(_self_close(_escape_amp(_wrap_root(s)))))
    try:
        ET.fromstring(s); return s
    except Exception:
        return None

# ---------- Rasterization ----------
def rasterize_svg(svg_str, size=224):
    if skia is not None:
        dom = skia.SVGDOM.MakeFromStream(skia.MemoryStream(svg_str.encode("utf-8")))
        if dom is None: raise RuntimeError("Skia could not parse SVG.")
        surface = skia.Surface(int(size), int(size)); canvas = surface.getCanvas()
        canvas.clear(skia.Color4f(1,1,1,0))
        try: dom.setContainerSize(skia.Size(float(size), float(size)))
        except TypeError: dom.setContainerSize(skia.Size.Make(float(size), float(size)))
        dom.render(canvas)
        img = surface.makeImageSnapshot()
        return Image.open(io.BytesIO(bytes(img.encodeToData()))).convert("RGB")
    elif cairosvg is not None:
        png = cairosvg.svg2png(bytestring=svg_str.encode("utf-8"), output_width=size, output_height=size)
        return Image.open(io.BytesIO(png)).convert("RGB")
    else:
        raise RuntimeError("No SVG rasterizer available.")

# ---------- Metrics ----------
def pil_to_tensor(img):
    arr = np.array(img).transpose(2,0,1)
    return torch.from_numpy(arr).float().unsqueeze(0)/255.0

_lpips_model = None
def get_lpips():
    global _lpips_model, lpips, torch
    if lpips is None or torch is None: return None
    if _lpips_model is None: _lpips_model = lpips.LPIPS(net="alex")
    return _lpips_model

def compute_ssim(x, y):
    if ssim_fn is None or torch is None: return None
    return float(ssim_fn(pil_to_tensor(x), pil_to_tensor(y), data_range=1.0))

def compute_lpips(x, y):
    if lpips is None or torch is None: return None
    loss = get_lpips()
    return float(loss(pil_to_tensor(x), pil_to_tensor(y)).item())

def token_edit_distance(a, b):
    tok = r"[A-Za-z]+|[-+]?\d*\.?\d+|#[0-9A-Fa-f]{3,6}|[<>/='\";:(),]"
    a_toks = re.findall(tok, a or ""); b_toks = re.findall(tok, b or "")
    na, nb = len(a_toks), len(b_toks)
    dp = [[0]*(nb+1) for _ in range(na+1)]
    for i in range(na+1): dp[i][0] = i
    for j in range(nb+1): dp[0][j] = j
    for i in range(1, na+1):
        ai = a_toks[i-1]
        for j in range(1, nb+1):
            cost = 0 if ai == b_toks[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[-1][-1], na, nb

class ClipScorer:
    def __init__(self, device="cpu", model="ViT-B-32", pt="laion2b_s34b_b79k"):
        self.available = (open_clip is not None and torch is not None)
        if not self.available: return
        self.device = torch.device(device)
        self.model, _, self.pre = open_clip.create_model_and_transforms(model, pretrained=pt)
        self.tok = open_clip.get_tokenizer(model)
        self.model.eval().to(self.device)
    def score(self, pil_img, text):
        if not self.available: return None
        with torch.no_grad():
            im = self.pre(pil_img).unsqueeze(0).to(self.device)
            tt = self.tok([text]).to(self.device)
            fi = self.model.encode_image(im); ft = self.model.encode_text(tt)
            fi = fi/fi.norm(dim=-1, keepdim=True); ft = ft/ft.norm(dim=-1, keepdim=True)
            return float((fi @ ft.T).squeeze().item())

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True)
    ap.add_argument("--metrics", default="raster_ssim,lpips,clipscore,token_edit")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    metrics = set(m.strip().lower() for m in args.metrics.split(",") if m.strip())
    with open(args.pred_jsonl, "r", encoding="utf-8", errors="replace") as f:
        preds = [json.loads(l) for l in f]

    avail = {"skia": skia is not None, "cairosvg": cairosvg is not None,
             "torch": torch is not None, "ssim_fn": ssim_fn is not None,
             "lpips": lpips is not None, "open_clip": open_clip is not None}
    if args.debug: print("Availability:", avail)

    clip_scorer = ClipScorer(device="cuda" if (torch and torch.cuda.is_available()) else "cpu")

    rows, failures, debug_msgs = [], {"raster":0,"ssim":0,"lpips":0,"clip":0}, []
    for idx, p in enumerate(preds):
        row = {}
        pred_svg = svg_autofix(p.get("svg_pred",""))
        gt_svg   = svg_autofix(p.get("svg_gt","")) if p.get("svg_gt") else None
        imgp = imgg = None

        if any(m in metrics for m in ("raster_ssim","lpips","clipscore")) and pred_svg:
            try:
                imgp = rasterize_svg(pred_svg, args.size)
                if ("raster_ssim" in metrics or "lpips" in metrics) and gt_svg:
                    imgg = rasterize_svg(gt_svg, args.size)
            except Exception as e:
                failures["raster"] += 1
                if args.debug and len(debug_msgs) < 8:
                    debug_msgs.append(f"[raster fail @ {idx}] {type(e).__name__}: {e}")

        if "raster_ssim" in metrics and imgp is not None and imgg is not None:
            try: row["ssim"] = compute_ssim(imgp, imgg)
            except Exception: failures["ssim"] += 1
        if "lpips" in metrics and imgp is not None and imgg is not None:
            try: row["lpips"] = compute_lpips(imgp, imgg)
            except Exception: failures["lpips"] += 1
        if "clipscore" in metrics and imgp is not None and clip_scorer and clip_scorer.available:
            try: row["clipscore"] = clip_scorer.score(imgp, p.get("caption",""))
            except Exception as e:
                failures["clip"] += 1
                if args.debug and len(debug_msgs) < 8:
                    debug_msgs.append(f"[clip fail @ {idx}] {type(e).__name__}: {e}")

        if "token_edit" in metrics:
            d, na, nb = token_edit_distance(p.get("svg_pred",""), p.get("svg_gt",""))
            row.update({"token_edit": d, "token_edit_norm": d/max(nb,1), "len_pred": na, "len_gt": nb})

        rows.append(row)

    keys = set().union(*(r.keys() for r in rows)) if rows else set()
    counts = {k: sum(1 for r in rows if r.get(k) is not None) for k in keys}
    agg = {k: float(np.mean([r[k] for r in rows if r.get(k) is not None])) for k in keys}
    print("Counts:", counts); print("Aggregate:", agg)
    if args.debug:
        print("Failures:", failures)
        for m in debug_msgs: print(m)

if __name__ == "__main__":
    main()
