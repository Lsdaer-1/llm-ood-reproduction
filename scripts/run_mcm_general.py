import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset_builder import build_dataset, get_class_names
from utils.metrics import summarize


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


def maybe_subsample(dataset, max_samples):
    if max_samples < 0:
        return dataset
    return Subset(dataset, list(range(min(max_samples, len(dataset)))))


@torch.no_grad()
def encode_text_features(model, processor, class_names, device):
    prompts = [f"a photo of a {name}" for name in class_names]

    all_feats = []

    for i in tqdm(range(0, len(prompts), 64), desc="Encoding class prompts"):
        batch = prompts[i:i + 64]
        inputs = processor(text=batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = model.get_text_features(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        feats = feats / feats.norm(dim=-1, keepdim=True)

        all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0)


@torch.no_grad()
def compute_mcm_scores(model, processor, loader, text_features, device):
    all_scores = []
    text_features = text_features.to(device)
    model.eval()

    for images, _ in tqdm(loader, desc="Computing MCM scores"):
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        out = model.get_image_features(pixel_values=pixel_values)
        image_features = out.pooler_output if hasattr(out, "pooler_output") else out
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = image_features @ text_features.T
        scores = logits.max(dim=1).values

        all_scores.append(scores.cpu().numpy())

    return np.concatenate(all_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=str(PROJECT_ROOT / "models" / "clip-vit-base-patch16"))
    parser.add_argument("--data_root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--id_dataset", default="tiny_imagenet")
    parser.add_argument("--ood_dataset", default="dtd")
    parser.add_argument("--id_split", default="val")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "mcm_general_results.json"))
    parser.add_argument("--ood_split", default="test")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading CLIP: {args.model_name}")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)

    print("Loading datasets...")
    id_dataset_full = build_dataset(args.id_dataset, args.data_root, split=args.id_split)
    ood_dataset_full = build_dataset(
        args.ood_dataset,
        args.data_root,
        split=args.ood_split,
    )

    class_names = get_class_names(id_dataset_full)

    id_dataset = maybe_subsample(id_dataset_full, args.max_samples)
    ood_dataset = maybe_subsample(ood_dataset_full, args.max_samples)

    print(f"ID dataset: {args.id_dataset}, samples={len(id_dataset)}, classes={len(class_names)}")
    print(f"OOD dataset: {args.ood_dataset}, samples={len(ood_dataset)}")
    print("First 10 ID classes:", class_names[:10])
    print("OOD split:", args.ood_split)
    print("OOD classes:", get_class_names(ood_dataset_full)[:20])
    print("OOD num classes:", len(get_class_names(ood_dataset_full)))

    id_loader = DataLoader(
        id_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pil,
    )

    ood_loader = DataLoader(
        ood_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pil,
    )

    text_features = encode_text_features(model, processor, class_names, device)

    print("Computing ID scores...")
    id_scores = compute_mcm_scores(model, processor, id_loader, text_features, device)

    print("Computing OOD scores...")
    ood_scores = compute_mcm_scores(model, processor, ood_loader, text_features, device)

    result = summarize(id_scores, ood_scores)
    result.update({
        "id_dataset": args.id_dataset,
        "ood_dataset": args.ood_dataset,
        "id_samples": len(id_scores),
        "ood_samples": len(ood_scores),
        "num_classes": len(class_names),
        "model_name": args.model_name,
    })

    print("\n========== MCM General Results ==========")
    print(f"ID dataset:  {args.id_dataset}")
    print(f"OOD dataset: {args.ood_dataset}")
    print(f"AUROC:       {result['auroc'] * 100:.2f}")
    print(f"FPR95:       {result['fpr95'] * 100:.2f}")
    print(f"ID mean:     {result['id_mean']:.4f}")
    print(f"OOD mean:    {result['ood_mean']:.4f}")
    print("========================================")

    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()