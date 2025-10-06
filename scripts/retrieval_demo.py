import argparse, json, os, re, math
from pathlib import Path
from collections import Counter, defaultdict

def tokenize(s: str):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return [w for w in s.split() if len(w) > 1]

def build_vocab(captions):
    df = Counter()
    for cap in captions:
        df.update(set(tokenize(cap)))
    return df

def compute_idf(df, n_docs):
    idf = {}
    for w, d in df.items():
        idf[w] = math.log((n_docs + 1) / (d + 1)) + 1.0
    return idf

def vectorize(text, idf):
    tf = Counter(tokenize(text))
    vec = {}
    for w, c in tf.items():
        if w in idf:
            vec[w] = c * idf[w]
    # L2 normalize
    norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
    for w in list(vec.keys()):
        vec[w] /= norm
    return vec

def cosine_sim(a, b):
    # a,b are sparse dicts
    if len(a) > len(b): a, b = b, a  # iterate over smaller
    return sum(a[w]*b.get(w, 0.0) for w in a)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", type=str, default="data/toy_text2svg.jsonl")
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--out", type=str, default="outputs/demo_output.svg")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    toy_path = root / args.toy
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(toy_path)]
    captions = [r["caption"] for r in rows]
    df = build_vocab(captions)
    idf = compute_idf(df, len(captions))
    query_vec = vectorize(args.query, idf)

    best_idx, best_sim = 0, -1.0
    for i, cap in enumerate(captions):
        cap_vec = vectorize(cap, idf)
        sim = cosine_sim(query_vec, cap_vec)
        if sim > best_sim:
            best_idx, best_sim = i, sim

    best_svg = rows[best_idx]["svg"]
    with open(out_path, "w") as f:
        f.write(best_svg)

    print(f"Query: {args.query}")
    print(f"Selected caption: {rows[best_idx]['caption']} (sim={best_sim:.3f})")
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
