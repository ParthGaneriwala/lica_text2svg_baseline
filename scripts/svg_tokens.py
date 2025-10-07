# scripts/svg_tokens.py
# Tiny SVG grammar: tokens for tags, attributes, path commands, and quantized numbers.
# Produces SHORT, valid-ish sequences; decoding rebuilds an SVG string.

from __future__ import annotations
import re
from typing import List, Tuple

# ---- Core vocabulary ----
TAGS_EMPTY = ["path", "rect", "circle", "ellipse", "line", "polyline", "polygon"]
TAGS_PAIR  = ["svg", "g", "defs", "clipPath", "linearGradient", "radialGradient"]
# we’ll treat empty tags as self-closing: <path/> etc.

ATTR_KEYS = [
    "d","fill","stroke","stroke-width","stroke-linecap","stroke-linejoin",
    "stroke-dasharray","opacity","fill-opacity","stroke-opacity",
    "x","y","cx","cy","rx","ry","r","x1","y1","x2","y2",
    "points","transform","viewBox","width","height","id","class"
]

PATH_CMDS = list("MLHVCSQTAZmlhvcsqtaz")  # supported path letters

# Numbers quantization: round to 2 decimals then bucket to string
def _quantize_num_str(s: str) -> str:
    try:
        v = float(s)
    except Exception:
        return s
    # clamp to a sane range and round (keeps vocabulary tiny but expressive)
    if abs(v) > 1e6:
        v = 0.0
    return f"{v:.2f}"

# Colors: keep #RGB/#RRGGBB; named colors pass through
HEX_RE = re.compile(r"^#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})$")

# Token constructors
def tok_tag_open(t: str) -> str:   return f"<{t}>"
def tok_tag_close(t: str) -> str:  return f"</{t}>"
def tok_empty(t: str) -> str:      return f"<{t}/>"
def tok_attr(k: str, v: str) -> str: return f"{k}={v}"

# Split a path "d" into commands and numbers; quantize numbers
def _tokenize_d(d: str) -> List[str]:
    out = []
    # Split commands and numbers; keep commands separate
    for part in re.findall(r"[MLHVCSQTAZmlhvcsqtaz]|[-+]?\d*\.?\d+(?:e[-+]?\d+)?|[, ]+", d):
        if part.strip() == "" or part.strip() == ",":
            continue
        if part in PATH_CMDS:
            out.append(f"d:{part}")
        else:
            out.append(f"d:{_quantize_num_str(part)}")
    return out

# Basic attribute tokenizer (numbers & lists quantized)
def _tokenize_attr(k: str, v: str) -> List[str]:
    k = k.strip()
    v = v.strip()
    if k == "d":
        return _tokenize_d(v)
    if k in ("points",):
        vals = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", v)
        return [tok_attr(k, _quantize_num_str(x)) for x in vals]
    if HEX_RE.match(v) or k in ("id","class"):
        return [tok_attr(k, v)]
    # numbers inside strings
    if re.search(r"[-+]?\d", v):
        parts = re.findall(r"[A-Za-z]+|[-+]?\d*\.?\d+(?:e[-+]?\d+)?|[#\w\-]+|[,() ]", v)
        toks = []
        buf = []
        for p in parts:
            if re.fullmatch(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", p):
                toks.append(tok_attr(k, _quantize_num_str(p)))
            elif p.strip() == "" or p == ",":
                continue
            else:
                # keep small words as literals (e.g., matrix, translate)
                toks.append(tok_attr(k, p))
        return toks or [tok_attr(k, v)]
    return [tok_attr(k, v)]

# Regexes to find tags/attrs
TAG_OPEN_RE  = re.compile(r"<\s*([A-Za-z_:][\w:.-]*)\b([^>]*)/?>")
TAG_CLOSE_RE = re.compile(r"</\s*([A-Za-z_:][\w:.-]*)\s*>")
ATTR_RE      = re.compile(r'([A-Za-z_:][\w:.-]*)\s*=\s*"([^"]*)"')

def tokenize_svg(svg: str) -> List[str]:
    svg = svg.strip()
    # Keep only from first tag (avoid stray text from LMs)
    first = svg.find("<")
    if first > 0: svg = svg[first:]
    tokens: List[str] = []
    pos = 0
    stack: List[str] = []

    while pos < len(svg):
        # closing tag?
        m_close = TAG_CLOSE_RE.match(svg, pos)
        if m_close:
            t = m_close.group(1)
            tokens.append(tok_tag_close(t))
            if stack and stack[-1] == t:
                stack.pop()
            pos = m_close.end()
            continue

        # opening/self-closing tag?
        m_open = TAG_OPEN_RE.match(svg, pos)
        if m_open:
            t = m_open.group(1)
            raw_attrs = m_open.group(2) or ""
            is_self = svg[m_open.end()-2:m_open.end()] == "/>"
            # tag token
            if t in TAGS_EMPTY or is_self:
                tokens.append(tok_empty(t))
            else:
                tokens.append(tok_tag_open(t))
                stack.append(t)

            # attributes
            for ak,av in ATTR_RE.findall(raw_attrs):
                ak = ak.strip()
                if ak not in ATTR_KEYS:
                    # keep unknown attrs but quantize numbers
                    tokens += _tokenize_attr(ak, av)
                else:
                    tokens += _tokenize_attr(ak, av)
            pos = m_open.end()
            continue

        # otherwise skip one char (robust to noise)
        pos += 1

    # ensure svg closed
    # (we do not force-close everything here; decoder will handle)
    return tokens

# ---- Decoding ----
def _attrs_from_tokens(tok_list: List[str]) -> List[str]:
    # merge consecutive same-key attrs by joining values (for points/d)
    attrs: dict[str, List[str]] = {}
    for t in tok_list:
        if t.startswith("d:"):
            attrs.setdefault("d", []).append(t[2:])
        elif "=" in t:
            k, v = t.split("=", 1)
            attrs.setdefault(k, []).append(v)
    out = []
    for k, vs in attrs.items():
        if k == "d":
            # stitch path: commands/numbers separated by spaces
            out.append(f'd="{" ".join(vs)}"')
        elif k == "points":
            # join as space-separated pairs if possible
            out.append(f'points="{" ".join(vs)}"')
        else:
            out.append(f'{k}="{" ".join(vs)}"')
    return out

def detokenize(tokens: List[str]) -> str:
    out: List[str] = []
    stack: List[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("</"):
            tag = t[2:-1]
            # pop stack until tag (robust)
            while stack and stack[-1] != tag:
                out.append(f"</{stack.pop()}>")
            if stack and stack[-1] == tag:
                stack.pop()
            out.append(f"</{tag}>")
            i += 1
            continue
        if t.startswith("<") and t.endswith("/>"):
            tag = t[1:-2]
            # collect following attrs for this tag (until next structural token)
            j = i+1
            attr_toks: List[str] = []
            while j < len(tokens) and ("<" not in tokens[j][:1]):
                attr_toks.append(tokens[j]); j += 1
            attrs = _attrs_from_tokens(attr_toks)
            out.append(f"<{tag} {' '.join(attrs)} />".replace("  ", " ").strip())
            i = j
            continue
        if t.startswith("<") and t.endswith(">"):
            tag = t[1:-1]
            stack.append(tag)
            j = i+1
            attr_toks: List[str] = []
            while j < len(tokens) and ("<" not in tokens[j][:1]):
                attr_toks.append(tokens[j]); j += 1
            attrs = _attrs_from_tokens(attr_toks)
            out.append(f"<{tag} {' '.join(attrs)}>".replace("  ", " ").strip())
            i = j
            continue
        # stray token → ignore (they should be attrs)
        i += 1

    # close any remaining
    while stack:
        out.append(f"</{stack.pop()}>")

    # ensure svg root present
    s = "".join(out)
    if "<svg" not in s:
        s = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">{s}</svg>'
    return s

# ---- Special tokens list for HF tokenizer ----
def special_tokens() -> List[str]:
    toks = []
    for t in TAGS_EMPTY: toks.append(tok_empty(t))
    for t in TAGS_PAIR:  toks += [tok_tag_open(t), tok_tag_close(t)]
    # attributes as k=VALUE tokens (VALUE resolved downstream); we add just the "k=" prefix to lock tokenization
    for k in ATTR_KEYS:
        toks.append(f"{k}=")
    # path tokens
    toks += [f"d:{c}" for c in PATH_CMDS]
    # We don't enumerate numbers; they’ll appear as plain text but already rounded to 2 decimals → tokenizer won’t explode vocab.
    return list(sorted(set(toks)))
