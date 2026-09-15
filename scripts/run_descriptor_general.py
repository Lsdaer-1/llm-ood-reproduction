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


def clean_descriptor(d):
    d = d.strip()
    d = d.lstrip("-•1234567890. ").strip()
    return d


def clean_descriptor_set(desc_set):
    return [
        clean_descriptor(d)
        for d in desc_set
        if clean_descriptor(d)
    ]


@torch.no_grad()
def encode_descriptor_features(model, processor, class_names, descriptors, device):
    all_class_text_features = []

    for cls in tqdm(class_names, desc="Encoding descriptor prompts"):
        desc_sets = descriptors[cls]

        prompts = []

        # fallback / class name prompt
        if len(desc_sets) == 0:
            prompts = [f"a photo of a {cls}"]
        else:
            for desc_set in desc_sets:
                for d in desc_set:
                    d = d.strip()
                    d = d.lstrip("-•1234567890. ").strip()

                    if not d:
                        continue

                    prompts.append(f"a photo of a {cls} which has {d}")

            if len(prompts) == 0:
                prompts = [f"a photo of a {cls}"]

        inputs = processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = model.get_text_features(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        feats = feats / feats.norm(dim=-1, keepdim=True)

        all_class_text_features.append(feats.cpu())

    return all_class_text_features

@torch.no_grad()
def compute_scores(model, processor, loader, text_features_by_class, device):
    all_scores = []
    model.eval()

    text_features_by_class = [
        feats.to(device) for feats in text_features_by_class
    ]

    for images, _ in tqdm(loader, desc="Computing descriptor scores"):
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        out = model.get_image_features(pixel_values=pixel_values)
        image_features = out.pooler_output if hasattr(out, "pooler_output") else out
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        class_scores = []

        for text_feats in text_features_by_class:
            sims = image_features @ text_feats.T
            score = sims.mean(dim=1)
            class_scores.append(score)

        class_scores = torch.stack(class_scores, dim=1)

        scores = class_scores.max(dim=1).values
        all_scores.append(scores.cpu().numpy())

    return np.concatenate(all_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=str(PROJECT_ROOT / "models" / "clip-vit-base-patch16"))
    parser.add_argument("--data_root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--id_dataset", default="oxford_pet")
    parser.add_argument("--id_split", default="test")
    parser.add_argument("--ood_dataset", default="ninco")
    parser.add_argument("--ood_split", default="hard")
    parser.add_argument("--descriptor_path", default=str(PROJECT_ROOT / "descriptors" / "oxford_pet_gpt_descriptors.json"))
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "descriptor_general_results.json"))
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading CLIP: {args.model_name}")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)

    print("Loading datasets...")
    id_dataset_full = build_dataset(args.id_dataset, args.data_root, split=args.id_split)
    ood_dataset_full = build_dataset(args.ood_dataset, args.data_root, split=args.ood_split)

    class_names = get_class_names(id_dataset_full)

    id_dataset = maybe_subsample(id_dataset_full, args.max_samples)
    ood_dataset = maybe_subsample(ood_dataset_full, args.max_samples)

    print(f"ID dataset: {args.id_dataset}, samples={len(id_dataset)}, classes={len(class_names)}")
    print(f"OOD dataset: {args.ood_dataset}, split={args.ood_split}, samples={len(ood_dataset)}")

    print(f"Loading descriptors: {args.descriptor_path}")
    descriptors = json.loads(Path(args.descriptor_path).read_text(encoding="utf-8"))

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

    text_features = encode_descriptor_features(
        model=model,
        processor=processor,
        class_names=class_names,
        descriptors=descriptors,
        device=device,
    )

    print("Computing ID scores...")
    id_scores = compute_scores(model, processor, id_loader, text_features, device)

    print("Computing OOD scores...")
    ood_scores = compute_scores(model, processor, ood_loader, text_features, device)

    result = summarize(id_scores, ood_scores)
    result.update({
        "id_dataset": args.id_dataset,
        "ood_dataset": args.ood_dataset,
        "ood_split": args.ood_split,
        "id_samples": len(id_scores),
        "ood_samples": len(ood_scores),
        "num_classes": len(class_names),
        "descriptor_path": args.descriptor_path,
    })

    print("\n========== Descriptor General Results ==========")
    print(f"ID dataset:  {args.id_dataset}")
    print(f"OOD dataset: {args.ood_dataset}")
    print(f"OOD split:   {args.ood_split}")
    print(f"AUROC:       {result['auroc'] * 100:.2f}")
    print(f"FPR95:       {result['fpr95'] * 100:.2f}")
    print(f"ID mean:     {result['id_mean']:.4f}")
    print(f"OOD mean:    {result['ood_mean']:.4f}")
    print("===============================================")

    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()