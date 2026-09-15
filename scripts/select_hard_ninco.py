import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset_builder import build_dataset, get_class_names


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate_pil(batch):
    images = [x[0] for x in batch]
    labels = [int(x[1]) for x in batch]
    return images, labels


@torch.no_grad()
def encode_text_features(model, processor, class_names, device):
    prompts = [f"a photo of a {name}" for name in class_names]

    inputs = processor(text=prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model.get_text_features(**inputs)
    feats = out.pooler_output if hasattr(out, "pooler_output") else out
    feats = feats / feats.norm(dim=-1, keepdim=True)

    return feats


@torch.no_grad()
def compute_image_mcm_scores(model, processor, loader, text_features, device):
    all_scores = []
    all_labels = []
    all_matches = []

    model.eval()

    for images, labels in tqdm(loader, desc="Scoring NINCO images"):
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        out = model.get_image_features(pixel_values=pixel_values)
        image_features = out.pooler_output if hasattr(out, "pooler_output") else out
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = image_features @ text_features.T

        scores, matches = logits.max(dim=1)

        all_scores.extend(scores.cpu().tolist())
        all_matches.extend(matches.cpu().tolist())
        all_labels.extend(labels)

    return all_scores, all_labels, all_matches


def main():
    random.seed(0)

    device = get_device()
    print(f"Using device: {device}")

    model_path = PROJECT_ROOT / "models" / "clip-vit-base-patch16"

    print(f"Loading CLIP: {model_path}")
    model = CLIPModel.from_pretrained(str(model_path)).to(device)
    processor = CLIPProcessor.from_pretrained(str(model_path))

    print("Loading Oxford Pet ID classes...")
    pet = build_dataset("oxford_pet", PROJECT_ROOT / "data", split="test")
    pet_classes = get_class_names(pet)

    print("Loading NINCO animal subset...")
    ninco = build_dataset("ninco", PROJECT_ROOT / "data", split="animal")
    ninco_classes = get_class_names(ninco)

    print(f"Oxford Pet classes: {len(pet_classes)}")
    print(f"NINCO animal classes: {len(ninco_classes)}")
    print(f"NINCO images: {len(ninco)}")

    print("Encoding Oxford Pet prompts...")
    text_features = encode_text_features(model, processor, pet_classes, device)

    loader = DataLoader(
        ninco,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pil,
    )

    scores, labels, matches = compute_image_mcm_scores(
        model=model,
        processor=processor,
        loader=loader,
        text_features=text_features,
        device=device,
    )

    by_class_scores = defaultdict(list)
    by_class_matches = defaultdict(list)

    for score, label, match in zip(scores, labels, matches):
        cls_name = ninco_classes[label]
        by_class_scores[cls_name].append(score)
        by_class_matches[cls_name].append(match)

    results = []

    for cls_name in ninco_classes:
        cls_scores = np.array(by_class_scores[cls_name])
        cls_matches = by_class_matches[cls_name]

        match_counts = defaultdict(int)
        for m in cls_matches:
            match_counts[m] += 1

        best_match_idx = max(match_counts.items(), key=lambda x: x[1])[0]
        best_match_name = pet_classes[best_match_idx]

        results.append({
            "class_name": cls_name,
            "num_images": int(len(cls_scores)),
            "mean_mcm": float(cls_scores.mean()),
            "std_mcm": float(cls_scores.std()),
            "max_mcm": float(cls_scores.max()),
            "closest_pet_class": best_match_name,
        })

    results.sort(key=lambda x: x["mean_mcm"], reverse=True)

    print("\n" + "=" * 100)
    print("Hard NINCO animal classes ranked by image-level MCM score")
    print("=" * 100)

    for r in results:
        print(
            f"{r['class_name']:35s} "
            f"mean={r['mean_mcm']:.4f} "
            f"std={r['std_mcm']:.4f} "
            f"max={r['max_mcm']:.4f} "
            f"n={r['num_images']:3d} "
            f"--> {r['closest_pet_class']}"
        )

    out = PROJECT_ROOT / "results" / "hard_ninco_by_image_scores.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()