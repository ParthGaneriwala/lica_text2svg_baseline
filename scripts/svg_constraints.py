# scripts/svg_constraints.py
from __future__ import annotations
from typing import List, Dict, Set
import torch
from transformers import LogitsProcessor

PAIR_TAGS  = ["svg","g","defs","clipPath","linearGradient","radialGradient"]
EMPTY_TAGS = ["path","rect","circle","ellipse","line","polyline","polygon"]

REQUIRED_ATTRS = {
    "path": ["d:"],  # path requires at least one d: token
    "rect": ["x=","y=","width","height"],
    "circle": ["cx","cy","r"],
    "ellipse": ["cx","cy","rx","ry"],
    "line": ["x1","y1","x2","y2"],
    "polyline": ["points"],
    "polygon": ["points"],
}

def is_open(tok: str) -> bool:
    return tok.startswith("<") and tok.endswith(">") and not tok.startswith("</")
def is_close(tok: str) -> bool:
    return tok.startswith("</") and tok.endswith(">")
def open_tag(tok: str) -> str:
    return tok[1:-1]
def close_tag(tok: str) -> str:
    return tok[2:-1]
def is_empty(tok: str) -> bool:
    return tok.startswith("<") and tok.endswith("/>")

def _structural(tok: str) -> bool:
    return tok.startswith("<")

def _has_drawable_so_far(seq: List[str]) -> bool:
    # “Drawable” if we’ve emitted any EMPTY tag (element) with at least one attribute afterwards
    # or an open tag followed by attributes.
    for i, t in enumerate(seq):
        if is_empty(t):
            # check that some attribute-like tokens follow before next structural token
            j = i+1
            while j < len(seq) and not _structural(seq[j]):
                if seq[j].startswith("d:") or "=" in seq[j]:
                    return True
                j += 1
        if is_open(t):
            # open tag, check attributes present after it
            j = i+1
            while j < len(seq) and not _structural(seq[j]):
                if seq[j].startswith("d:") or "=" in seq[j]:
                    return True
                j += 1
    return False

def _since_last_element_attrs(seq: List[str]) -> Set[str]:
    # Return a set of "attr hints" since the most recent structural element
    attrs: Set[str] = set()
    i = len(seq) - 1
    # Find last structural (tag) index
    while i >= 0 and not _structural(seq[i]):
        i -= 1
    j = i + 1
    while j < len(seq) and not _structural(seq[j]):
        if seq[j].startswith("d:"):
            attrs.add("d:")
        elif "=" in seq[j]:
            k = seq[j].split("=",1)[0]
            attrs.add(k)
        j += 1
    return attrs

class SVGConstrainedProcessor(LogitsProcessor):
    def __init__(self, id2tok: Dict[int,str], tok2id: Dict[str,int], min_struct_tokens: int = 12):
        super().__init__()
        self.id2tok = id2tok
        self.tok2id = tok2id
        self.min_struct = min_struct_tokens  # delay </svg> until some structure exists

    def _step_rules(self, seq_tokens: List[str]) -> set[str]:
        # Build stack and count structural tokens
        stack: List[str] = []
        struct_count = 0
        for t in seq_tokens:
            if _structural(t):
                struct_count += 1
            if is_open(t):
                tag = open_tag(t)
                if tag in PAIR_TAGS:
                    stack.append(tag)
            elif is_close(t):
                tag = close_tag(t)
                if stack and stack[-1] == tag:
                    stack.pop()

        allowed: set[str] = set()

        # First structural token must be <svg>
        if not any(_structural(t) for t in seq_tokens):
            allowed.add("<svg>")
            return allowed

        # Always allow attribute/value-ish tokens
        for tok in self.id2tok.values():
            if tok.endswith("=") or tok.startswith("d:"):
                allowed.add(tok)

        # Determine last element and enforce minimal attrs before new element/closing
        last_elem = None
        for i in range(len(seq_tokens)-1, -1, -1):
            if is_empty(seq_tokens[i]) or is_open(seq_tokens[i]):
                last_elem = seq_tokens[i]
                break
        needed = set()
        if last_elem is not None:
            tag = open_tag(last_elem) if is_open(last_elem) else last_elem[1:-2]
            reqs = REQUIRED_ATTRS.get(tag, [])
            have = _since_last_element_attrs(seq_tokens)
            for r in reqs:
                if r not in have:
                    needed.add(r)

        # Allow opening EMPTY tags and PAIR tags
        for t in EMPTY_TAGS:
            # If we still owe required attrs for the current element, do not allow a new element yet
            if not needed:
                allowed.add(f"<{t}/>")
        for t in PAIR_TAGS:
            if not needed:
                allowed.add(f"<{t}>")

        # Allow the correct closing tag only (if stack non-empty)
        if stack:
            # only close the top if no pending required attrs
            if not needed:
                allowed.add(f"</{stack[-1]}>")
        else:
            # Only allow </svg> if min structure reached and we have at least one drawable element
            if struct_count >= self.min_struct and _has_drawable_so_far(seq_tokens) and not needed:
                allowed.add("</svg>")

        return allowed

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch, vocab = scores.shape
        special_ids = set(self.tok2id.values())
        for b in range(batch):
            toks = [self.id2tok.get(int(t), "") for t in input_ids[b].tolist()]
            toks = [t for t in toks if t in self.tok2id]  # only our special tokens
            allowed = self._step_rules(toks)
            # Mask special tokens not allowed; leave non-special logits untouched
            mask = torch.zeros_like(scores[b])
            for tid, tok in self.id2tok.items():
                if tid in special_ids and tok not in allowed:
                    mask[tid] = float('-inf')
            scores[b] = scores[b] + mask
        return scores
