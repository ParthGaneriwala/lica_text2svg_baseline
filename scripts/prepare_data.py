import argparse, json, random
from datasets import load_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="text2svg-stack", choices=["text2svg-stack", "svg-stack-simple"])
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--sample", type=int, default=50000)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed)

    if args.dataset == "text2svg-stack":
        ds = load_dataset("starvector/text2svg-stack", split=args.split)
        # prefer LLaVA/CogVLM/BLIP2 captions if present, fallback to any available
        def pick_caption(r):
            for k in ["caption_llava", "caption_cogvlm", "caption_blip2"]:
                if r.get(k): return r[k]
            return ""
        rows = [{"caption": pick_caption(r), "svg": r["svg"]} for r in ds]
        rows = [r for r in rows if r["caption"] and r["svg"]]
    else:
        ds = load_dataset("starvector/svg-stack-simple", split=args.split)
        rows = [{"caption": "", "svg": r["svg"]} for r in ds]

    if args.sample and args.sample < len(rows):
        rows = random.sample(rows, args.sample)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
