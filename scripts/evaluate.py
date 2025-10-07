import argparse, json, io, os, re
from pathlib import Path
from PIL import Image
import numpy as np


try:
    import skia  # pip install skia-python
except Exception:
    skia = None
try:
    import cairosvg  # pip install cairosvg (needs system cairo on Windows)
except Exception:
    cairosvg = None

try:
    import torch
except Exception:
    torch = None
try:
    # torchmetrics function name can vary across versions
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

# -------------------- SVG AUTO-FIX HELPERS --------------------
import xml.etree.ElementTree as ET

_ALLOWED_EMPTY = {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "stop"}
_SVG_ROOT_TMPL = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256">{content}</svg>'

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s)
    s = s.replace("<pad>", "").replace("</s>", "").replace("<s>", "")
    s = re.sub(r"<extra_id_\d+>", "", s)
    return s.strip()

def _quote_attrs(s: str) -> str:
    # turn foo=bar into foo="bar" (only where value has no spaces or quotes)
    return re.sub(r'(\s[\w:-]+)=([#\w\.\-+,/]+)(?=[\s>/])', r'\1="\2"', s)

def _self_close_empty_tags(s: str) -> str:
    # <path ...> -> <path .../> if there is no explicit closing tag
    def repl(m):
        tag = m.group(1).lower()
        attrs = m.group(2)
        if tag in _ALLOWED_EMPTY:
            return f"<{tag}{attrs}/>"
        return m.group(0)
    return re.sub(r"<(\w+)([^>/]*?)>(?!\s*</\1>)", repl, s)

def _keep_only_svg_tags(s: str) -> str:
    # crude sanitizer: drop script/foreignObject and unexpected tags
    s = re.sub(r"</?(script|foreignObject|iframe)[^>]*>", "", s, flags=re.I)
    return s

def _wrap_if_no_root(s: str) -> str:
    if "<svg" in s.lower():
        return s
    frag = s[s.find("<"): s.rfind(">") + 1] if "<" in s and ">" in s else s
    return _SVG_ROOT_TMPL.format(content=frag)

def _escape_amp(s: str) -> str:
    # escape & that aren’t entities
    return re.sub(r"&(?![a-zA-Z#0-9]+;)", "&amp;", s)

def svg_autofix(raw: str):
    """Return a best-effort valid SVG string or None if unrecoverable."""
    if not raw or not isinstance(raw, str):
        return None
    s = _strip_code_fences(raw)
    # keep from first SVG-ish tag
    m = re.search(r"<(svg|path|rect|circle|polygon|polyline|line|g|defs|linearGradient|radialGradient)\b.*", s, flags=re.I|re.S)
    if m:
        s = s[m.start():]
    s = _keep_only_svg_tags(s)
    s = _quote_attrs(s)
    s = _self_close_empty_tags(s)
    s = _escape_amp(s)
    s = _wrap_if_no_root(s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    try:
        ET.fromstring(s)
        return s
    except Exception:
        return None

# -------------------- RASTERIZATION --------------------
def rasterize_svg(svg_str, size=224):
    """Return a PIL.Image rendered from an SVG string at a given square size."""
    if skia is not None:
        data = svg_str.encode("utf-8")
        stream = skia.MemoryStream(data)
        dom = skia.SVGDOM.MakeFromStream(stream)
        if dom is None:
            raise RuntimeError("Skia could not parse SVG.")

        # Create surface & canvas
        surface = skia.Surface(int(size), int(size))
        canvas = surface.getCanvas()
        canvas.clear(skia.Color4f(1, 1, 1, 0))  # transparent

        # IMPORTANT: set container size using skia.Size
        try:
            dom.setContainerSize(skia.Size(float(size), float(size)))
        except TypeError:
            # older/newer bindings sometimes expose Make()
            dom.setContainerSize(skia.Size.Make(float(size), float(size)))

        # Render
        dom.render(canvas)

        img = surface.makeImageSnapshot()
        data = img.encodeToData()  # PNG
        return Image.open(io.BytesIO(bytes(data))).convert("RGB")

    elif cairosvg is not None:
        png_bytes = cairosvg.svg2png(bytestring=svg_str.encode("utf-8"),
                                     output_width=size, output_height=size)
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")
    else:
        raise RuntimeError("No SVG rasterizer available. Install skia-python (recommended) or cairosvg.")


# -------------------- METRICS --------------------
def pil_to_tensor(img):
    arr = np.array(img).transpose(2, 0, 1)  # CHW
    return torch.from_numpy(arr).float().unsqueeze(0) / 255.0

def compute_ssim(x_img, y_img):
    if ssim_fn is None or torch is None:
        return None
    return float(ssim_fn(pil_to_tensor(x_img), pil_to_tensor(y_img), data_range=1.0))

def compute_lpips(x_img, y_img):
    if lpips is None or torch is None:
        return None
    loss_fn = lpips.LPIPS(net="alex")
    return float(loss_fn(pil_to_tensor(x_img), pil_to_tensor(y_img)).item())

def token_edit_distance(a, b):
    # simple SVG-aware tokenization: tags/attrs/nums/colors/punct
    a_toks = re.findall(r"[A-Za-z]+|[-+]?\d*\.?\d+|#[0-9A-Fa-f]{3,6}|[<>/='\";:(),]", a)
    b_toks = re.findall(r"[A-Za-z]+|[-+]?\d*\.?\d+|#[0-9A-Fa-f]{3,6}|[<>/='\";:(),]", b)
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
    def __init__(self, device="cpu", model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", image_size=224):
        self.available = (open_clip is not None and torch is not None)
        if not self.available:
            return
        self.device = torch.device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval().to(self.device)
        self.image_size = image_size

    def score(self, pil_img, text: str):
        if not self.available:
            return None
        with torch.no_grad():
            image = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            text_tokens = self.tokenizer([text]).to(self.device)
            img_feat = self.model.encode_image(image)
            txt_feat = self.model.encode_text(text_tokens)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat @ txt_feat.T).squeeze().item()
        return float(sim)  # cosine similarity in [-1,1]

# -------------------- MAIN --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", type=str, required=True,
                    help="predictions.jsonl with fields: caption, svg_pred, svg_gt")
    ap.add_argument("--metrics", type=str, default="raster_ssim,lpips,clipscore,token_edit")
    ap.add_argument("--size", type=int, default=224, help="raster size for metrics")
    ap.add_argument("--debug", action="store_true", help="print availability and first few failures")
    args = ap.parse_args()

    metrics = set(m.strip().lower() for m in args.metrics.split(",") if m.strip())

    # robust UTF-8 read (Windows-safe)
    with open(args.pred_jsonl, "r", encoding="utf-8", errors="replace") as f:
        preds = [json.loads(l) for l in f]

    # availability snapshot
    avail = {
        "skia": skia is not None,
        "cairosvg": cairosvg is not None,
        "torch": torch is not None,
        "ssim_fn": ssim_fn is not None,
        "lpips": lpips is not None,
        "open_clip": open_clip is not None,
    }
    if args.debug:
        print("Availability:", avail)

    clip_scorer = None
    if "clipscore" in metrics:
        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        clip_scorer = ClipScorer(device=device, image_size=args.size)

    rows = []
    failures = {"raster": 0, "ssim": 0, "lpips": 0, "clip": 0}
    debug_msgs = []

    for idx, p in enumerate(preds):
        row = {}
        img_pred = img_gt = None

        # try to repair SVGs first
        pred_svg = svg_autofix(p.get("svg_pred", ""))
        gt_svg   = svg_autofix(p.get("svg_gt", "")) if p.get("svg_gt") else None

        if any(m in metrics for m in ("raster_ssim", "lpips", "clipscore")) and pred_svg is not None:
            try:
                img_pred = rasterize_svg(pred_svg, size=args.size)
                if ("raster_ssim" in metrics or "lpips" in metrics) and gt_svg is not None:
                    img_gt = rasterize_svg(gt_svg, size=args.size)
            except Exception as e:
                failures["raster"] += 1
                if args.debug and len(debug_msgs) < 5:
                    debug_msgs.append(f"[raster fail @ {idx}] {type(e).__name__}: {e}")

        if "raster_ssim" in metrics and img_pred is not None and img_gt is not None:
            try:
                row["ssim"] = compute_ssim(img_pred, img_gt)
            except Exception:
                failures["ssim"] += 1

        if "lpips" in metrics and img_pred is not None and img_gt is not None:
            try:
                row["lpips"] = compute_lpips(img_pred, img_gt)
            except Exception:
                failures["lpips"] += 1

        if "token_edit" in metrics:
            dist, na, nb = token_edit_distance(p.get("svg_pred", ""), p.get("svg_gt", ""))
            row["token_edit"] = dist
            row["len_pred"] = na
            row["len_gt"] = nb
            row["token_edit_norm"] = dist / max(nb, 1)

        if "clipscore" in metrics and img_pred is not None and clip_scorer is not None and clip_scorer.available:
            try:
                row["clipscore"] = clip_scorer.score(img_pred, p.get("caption", ""))
            except Exception as e:
                failures["clip"] += 1
                if args.debug and len(debug_msgs) < 5:
                    debug_msgs.append(f"[clip fail @ {idx}] {type(e).__name__}: {e}")

        rows.append(row)

    # coverage + aggregates
    keys = set().union(*(r.keys() for r in rows)) if rows else set()
    counts = {k: sum(1 for r in rows if k in r and r[k] is not None) for k in keys}
    agg = {k: float(np.mean([r[k] for r in rows if r.get(k) is not None])) for k in keys if any(r.get(k) is not None for r in rows)}

    print("Counts:", counts)
    print("Aggregate:", agg)
    if args.debug:
        print("Failures:", failures)
        for m in debug_msgs:
            print(m)

if __name__ == "__main__":
    main()
