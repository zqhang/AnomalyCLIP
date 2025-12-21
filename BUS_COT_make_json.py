import json
import argparse
from pathlib import Path


def ensure_two_image_prompt(q: str) -> str:
    # replace first <image> with <image><image> if needed
    if "<image><image>" in q:
        return q
    if "<image>" in q:
        return q.replace("<image>", "<image><image>", 1)
    # if prompt has no tag, append
    return q + "\n<image><image>\n"


def main():
    parser = argparse.ArgumentParser("Create 2-image JSON: [original, anomaly_map]")
    parser.add_argument("--input_json", required=True, type=str)
    parser.add_argument("--out_json", required=True, type=str)
    parser.add_argument("--heatmap_dir", required=True, type=str)
    parser.add_argument("--heatmap_prefix", default="anomaly_map_", type=str)
    parser.add_argument("--note", default="Second image is an anomaly heatmap (hotter = more suspicious).", type=str)
    args = parser.parse_args()

    heat_dir = Path(args.heatmap_dir)

    items = json.load(open(args.input_json, "r", encoding="utf-8"))
    out = []

    missing = 0
    for it in items:
        orig = it["images"][0]
        orig_name = Path(orig).name

        heat_path = heat_dir / f"{args.heatmap_prefix}{orig_name}"

        if not heat_path.exists():
            missing += 1

        new_it = dict(it)
        new_it["images"] = [orig, str(heat_path)]

        q = new_it.get("query", "")
        q = ensure_two_image_prompt(q)
        if args.note and args.note not in q:
            q = q.rstrip() + "\n" + args.note + "\n"
        new_it["query"] = q

        out.append(new_it)

    json.dump(out, open(args.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Wrote {len(out)} samples -> {args.out_json}")
    if missing:
        print(f"WARNING: {missing} heatmaps missing (paths will still be written). Check heatmap_dir/prefix.")


if __name__ == "__main__":
    main()
