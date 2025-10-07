#!/usr/bin/env python
import argparse, json, re, numpy as np

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True)
    args = ap.parse_args()
    with open(args.pred_jsonl, "r", encoding="utf-8", errors="replace") as f:
        rows = [json.loads(l) for l in f]
    out = []
    for r in rows:
        d, na, nb = token_edit_distance(r.get("svg_pred",""), r.get("svg_gt",""))
        out.append({"token_edit": d, "token_edit_norm": d/max(nb,1), "len_pred": na, "len_gt": nb})
    keys = out[0].keys() if out else []
    counts = {k: sum(1 for x in out if x.get(k) is not None) for k in keys}
    agg = {k: float(np.mean([x[k] for x in out if x.get(k) is not None])) for k in keys}
    print("Counts:", counts)
    print("Aggregate:", agg)

if __name__ == "__main__":
    main()
