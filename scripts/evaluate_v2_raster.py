#!/usr/bin/env python
import argparse, io, json, re, numpy as np
from PIL import Image

# raster backends
try:
    import skia
except Exception:
    skia = None
try:
    import cairosvg
except Exception:
    cairosvg = None

# metrics
try:
    import torch
except Exception:
    torch = None
try:
    from torchmetrics.functional.image.ssim import structural_similarity_index_measure as ssim_fn
except Exception:
    ssim_fn = None
try:
    import lpips
except Exception:
    lpips = None
try:
    import open_clip
except Exception:
    open_clip = None

def rasterize(svg_str, size=224):
    if skia is not None:
        dom = skia.SVGDOM.MakeFromStream(skia.MemoryStream(svg_str.encode("utf-8")))
        if dom is None: raise RuntimeError("Skia parse fail")
        surf = skia.Surface(size, size); can = surf.getCanvas()
        can.clear(skia.Color4f(1,1,1,0))
        try: dom.setContainerSize(skia.Size(float(size), float(size)))
        except TypeError: dom.setContainerSize(skia.Size.Make(float(size), float(size)))
        dom.render(can)
        img = surf.makeImageSnapshot()
        return Image.open(io.BytesIO(bytes(img.encodeToData()))).convert("RGB")
    if cairosvg is not None:
        png = cairosvg.svg2png(bytestring=svg_str.encode("utf-8"), output_width=size, output_height=size)
        return Image.open(io.BytesIO(png)).convert("RGB")
    raise RuntimeError("No rasterizer: install skia-python or cairosvg")

def pil2t(img):
    import numpy as np, torch
    return torch.from_numpy(np.array(img).transpose(2,0,1)).float().unsqueeze(0)/255.0

def token_edit_distance(a, b):
    tok = r"[A-Za-z]+|[-+]?\d*\.?\d+|#[0-9A-Fa-f]{3,6}|[<>/='\";:(),]"
    a_toks = re.findall(tok, a or "")
    b_toks = re.findall(tok, b or "")
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
        self.ok = (open_clip is not None and torch is not None)
        if not self.ok: return
        self.device = torch.device(device)
        self.model, _, self.pre = open_clip.create_model_and_transforms(model, pretrained=pt)
        self.tok = open_clip.get_tokenizer(model)
        self.model.eval().to(self.device)
    def score(self, img, text):
        if not self.ok: return None
        with torch.no_grad():
            im = self.pre(img).unsqueeze(0).to(self.device)
            tt = self.tok([text]).to(self.device)
            fi = self.model.encode_image(im); ft = self.model.encode_text(tt)
            fi = fi/fi.norm(dim=-1, keepdim=True); ft = ft/ft.norm(dim=-1, keepdim=True)
            return float((fi@ft.T).squeeze().item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True)
    ap.add_argument("--metrics", default="raster_ssim,lpips,clipscore,token_edit")
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()
    metrics = set(x.strip() for x in args.metrics.split(",") if x.strip())
    with open(args.pred_jsonl, "r", encoding="utf-8", errors="replace") as f:
        rows = [json.loads(l) for l in f]
    clip = ClipScorer(device="cuda" if (torch and torch.cuda.is_available()) else "cpu")
    out = []
    for r in rows:
        row = {}
        imgp = imgg = None
        if any(m in metrics for m in ("raster_ssim","lpips","clipscore")):
            try:
                imgp = rasterize(r.get("svg_pred",""), args.size)
                if "raster_ssim" in metrics or "lpips" in metrics:
                    imgg = rasterize(r.get("svg_gt",""), args.size)
            except Exception:
                imgp = imgg = None
        if "raster_ssim" in metrics and imgp is not None and imgg is not None and ssim_fn and torch:
            row["ssim"] = float(ssim_fn(pil2t(imgp), pil2t(imgg), data_range=1.0))
        if "lpips" in metrics and imgp is not None and imgg is not None and lpips and torch:
            row["lpips"] = float(lpips.LPIPS(net="alex")(pil2t(imgp), pil2t(imgg)).item())
        if "clipscore" in metrics and imgp is not None:
            row["clipscore"] = clip.score(imgp, r.get("caption",""))
        if "token_edit" in metrics:
            d, na, nb = token_edit_distance(r.get("svg_pred",""), r.get("svg_gt",""))
            row.update({"token_edit": d, "token_edit_norm": d/max(nb,1), "len_pred": na, "len_gt": nb})
        out.append(row)
    keys = set().union(*(o.keys() for o in out)) if out else set()
    counts = {k: sum(1 for o in out if k in o and o[k] is not None) for k in keys}
    agg = {k: float(np.mean([o[k] for o in out if o.get(k) is not None])) for k in keys}
    print("Counts:", counts); print("Aggregate:", agg)

if __name__ == "__main__":
    main()
