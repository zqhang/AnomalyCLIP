import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
import cv2

import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner
from utils import get_transform, normalize


def apply_ad_scoremap(image_rgb_uint8, scoremap_0_1, alpha=0.5):
    """
    image_rgb_uint8: HxWx3 uint8
    scoremap_0_1: HxW float in [0,1]
    returns: HxWx3 uint8
    """
    np_image = image_rgb_uint8.astype(np.float32)
    sm = (scoremap_0_1 * 255).astype(np.uint8)
    sm = cv2.applyColorMap(sm, cv2.COLORMAP_JET)
    sm = cv2.cvtColor(sm, cv2.COLOR_BGR2RGB).astype(np.float32)
    out = alpha * np_image + (1.0 - alpha) * sm
    return np.clip(out, 0, 255).astype(np.uint8)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Generate AnomalyCLIP heatmaps fast (load once)")
    parser.add_argument("--input_json", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--checkpoint_path", required=True, type=str)

    # match your current defaults
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--sigma", type=int, default=4)

    # speed knobs
    parser.add_argument("--fp16", action="store_true", help="Use autocast fp16 (faster on GPU)")
    parser.add_argument("--save_mode", choices=["overlay", "raw", "both"], default="overlay",
                        help="overlay=colored overlay, raw=grayscale heatmap, both=save both")

    # make sure CLIP cache is writable
    parser.add_argument("--clip_cache", type=str, default=os.path.expanduser("~/.cache/clip"))

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.clip_cache, exist_ok=True)

    # --- load JSON and unique image paths ---
    with open(args.input_json, "r", encoding="utf-8") as f:
        items = json.load(f)
    img_paths = sorted({it["images"][0] for it in items})
    print(f"Found {len(items)} samples, {len(img_paths)} unique images.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- build model ONCE ---
    AnomalyCLIP_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
    }

    print("Loading backbone once...")
    model, _ = AnomalyCLIP_lib.load(
        "ViT-L/14@336px",
        device=device,
        design_details=AnomalyCLIP_parameters,
        download_root=args.clip_cache,
    )
    model.eval()

    preprocess, _ = get_transform(args)

    print("Loading prompt learner checkpoint once...")
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)

    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    # your checkpoint stores prompt_learner
    prompt_learner.load_state_dict(ckpt["prompt_learner"])

    prompt_learner.to(device)
    model.to(device)

    # required by AnomalyCLIP
    model.visual.DAPM_replace(DPAM_layer=20)

    # precompute text features ONCE
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # optional: speed on GPU
    use_autocast = args.fp16 and (device == "cuda")

    # --- loop images ---
    for i, image_path in enumerate(img_paths, 1):
        p = Path(image_path)
        out_overlay = Path(args.out_dir) / f"anomaly_map_{p.name}"
        out_raw = Path(args.out_dir) / f"anomaly_raw_{p.stem}.png"

        # skip if already exists (nice for resume)
        if args.save_mode in ("overlay", "both") and out_overlay.exists():
            if args.save_mode == "overlay":
                print(f"[{i}/{len(img_paths)}] skip (exists): {p.name}")
                continue
        if args.save_mode in ("raw", "both") and out_raw.exists():
            if args.save_mode == "raw":
                print(f"[{i}/{len(img_paths)}] skip (exists): {p.name}")
                continue

        print(f"[{i}/{len(img_paths)}] {p.name}")

        img_pil = Image.open(str(p)).convert("RGB")
        img = preprocess(img_pil)  # tensor CxHxW
        image = img.reshape(1, 3, args.image_size, args.image_size).to(device)

        if use_autocast:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)
        else:
            image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # (image-level prob exists but we only need map)
        # text_probs = (image_features @ text_features.permute(0, 2, 1) / 0.07).softmax(-1)[:, 0, 1]

        anomaly_map_list = []
        for idx_pf, patch_feature in enumerate(patch_features):
            if idx_pf >= args.feature_map_layer[0]:
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                #similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])

                pf = patch_feature.float()
                tf = text_features[0].float()
                similarity, _ = AnomalyCLIP_lib.compute_similarity(pf, tf)


                similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                anomaly_map_list.append(anomaly_map)

        anomaly_map = torch.stack(anomaly_map_list).sum(dim=0)  # [1, H, W] typically
        # gaussian smoothing (same as your code)
        anomaly_map = torch.stack(
            [torch.from_numpy(gaussian_filter(m, sigma=args.sigma)) for m in anomaly_map.detach().cpu()],
            dim=0,
        )  # still [1,H,W]

        # normalize to 0..1 for saving
        am = anomaly_map[0].numpy().astype(np.float32)
        am = normalize(am)  # your utils.normalize -> 0..1

        # save raw heatmap (grayscale)
        if args.save_mode in ("raw", "both"):
            raw_u8 = (am * 255).astype(np.uint8)
            cv2.imwrite(str(out_raw), raw_u8)

        # save overlay (colored) resized to img_size
        if args.save_mode in ("overlay", "both"):
            # load original image in cv2 and resize to match am
            base = cv2.imread(str(p))
            base = cv2.resize(base, (args.image_size, args.image_size))
            base = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
            overlay = apply_ad_scoremap(base, am, alpha=0.5)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_overlay), overlay)

    print("Done.")


if __name__ == "__main__":
    main()
