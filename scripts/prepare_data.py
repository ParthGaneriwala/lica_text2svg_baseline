import argparse, json, random
from datasets import load_dataset

def pick_caption(r):
    for k in ["caption_llava", "caption_cogvlm", "caption_blip2", "caption"]:
        if r.get(k):
            return r[k]
    return ""

def pick_svg(r):
    # Be robust to different dataset schemas
    for k in ["Svg", "svg", "SVG"]:
        if r.get(k):
            return r[k]
    raise KeyError(f"No SVG field found in row; available keys: {list(r.keys())[:20]}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="text2svg-stack",
                    choices=["text2svg-stack", "svg-stack-simple", "svg-stack", "text-to-svg"])
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--sample", type=int, default=50000)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed)

    if args.dataset == "text2svg-stack":
        ds = load_dataset("starvector/text2svg-stack", split=args.split)
        rows = [{"caption": pick_caption(r), "svg": pick_svg(r)} for r in ds]
        rows = [r for r in rows if r["caption"] and r["svg"]]
    elif args.dataset == "svg-stack-simple":
        ds = load_dataset("starvector/svg-stack-simple", split=args.split)
        rows = [{"caption": "", "svg": pick_svg(r)} for r in ds]
    elif args.dataset == "svg-stack":
        ds = load_dataset("starvector/svg-stack", split=args.split)
        rows = [{"caption": "", "svg": pick_svg(r)} for r in ds]
    elif args.dataset == "text-to-svg":
        # Example: wexhi/text-to-svg uses lowercase 'svg' + 'caption'
        ds = load_dataset("wexhi/text-to-svg", split=args.split)
        rows = [{"caption": pick_caption(r), "svg": pick_svg(r)} for r in ds]
        rows = [r for r in rows if r["caption"] and r["svg"]]
    else:
        raise ValueError("Unknown dataset option")

    if args.sample and args.sample < len(rows):
        rows = random.sample(rows, args.sample)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
