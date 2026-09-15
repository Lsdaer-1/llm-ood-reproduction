import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.metrics import summarize


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataset(name: str, root: str):
    if name == "cifar10":
        return CIFAR10(root=root, train=False, download=True, transform=None)
    if name == "cifar100":
        return CIFAR100(root=root, train=False, download=True, transform=None)
    if name == "svhn":
        return SVHN(root=root, split="test", download=True, transform=None)
    raise ValueError(name)


def maybe_subsample(dataset, max_samples):
    if max_samples < 0:
        return dataset
    return Subset(dataset, list(range(min(max_samples, len(dataset)))))


def collate_pil(batch) -> Tuple[List[Image.Image], List[int]]:
    return [x[0] for x in batch], [int(x[1]) for x in batch]


def load_descriptors(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))

    cleaned = {}
    for cls, descs in data.items():
        cleaned_descs = []
        for d in descs:
            d = d.strip()
            if not d:
                continue
            if d.lower().startswith("there are several useful visual features"):
                continue
            cleaned_descs.append(d)
        cleaned[cls] = cleaned_descs

    return cleaned


def build_text_prompts(class_names, descriptors):
    prompts_by_class = []

    for cls in class_names:
        prompts = [f"a photo of a {cls}"]

        for d in descriptors.get(cls, []):
            prompts.append(f"a photo of a {cls} which has {d}")

        prompts_by_class.append(prompts)

    return prompts_by_class


@torch.no_grad()
def encode_prompts(model, processor, prompts_by_class, device):
    class_features = []

    for cls_prompts in prompts_by_class:
        inputs = processor(text=cls_prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = model.get_text_features(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        feats = feats / feats.norm(dim=-1, keepdim=True)

        class_features.append(feats)

    return class_features


@torch.no_grad()
def compute_scores(model, processor, loader, class_features, device):
    all_scores = []
    model.eval()

    for images, _ in tqdm(loader, desc="Computing descriptor scores"):
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        out = model.get_image_features(pixel_values=pixel_values)
        image_features = out.pooler_output if hasattr(out, "pooler_output") else out
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        class_scores = []

        for feats in class_features:
            sims = image_features @ feats.T
            score = sims.mean(dim=1)
            class_scores.append(score)

        class_scores = torch.stack(class_scores, dim=1)
        final_scores = class_scores.max(dim=1).values

        all_scores.append(final_scores.cpu().numpy())

    return np.concatenate(all_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="../models/clip-vit-base-patch16")
    parser.add_argument("--descriptor_path", default="../descriptors/cifar10_gpt_descriptors.json")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--ood_dataset", default="cifar100", choices=["cifar100", "svhn"])
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output", default="./results/descriptor_cifar_results.json")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading CLIP: {args.model_name}")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)

    print(f"Loading descriptors: {args.descriptor_path}")
    descriptors = load_descriptors(Path(args.descriptor_path))

    prompts_by_class = build_text_prompts(CIFAR10_CLASSES, descriptors)

    print("\nExample prompts for dog:")
    for p in prompts_by_class[CIFAR10_CLASSES.index("dog")]:
        print("  ", p)

    print("\nEncoding descriptor prompts...")
    class_features = encode_prompts(model, processor, prompts_by_class, device)

    print("Loading datasets...")
    id_dataset = maybe_subsample(build_dataset("cifar10", args.data_root), args.max_samples)
    ood_dataset = maybe_subsample(build_dataset(args.ood_dataset, args.data_root), args.max_samples)

    id_loader = DataLoader(id_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pil)
    ood_loader = DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pil)

    print("Computing ID scores...")
    id_scores = compute_scores(model, processor, id_loader, class_features, device)

    print("Computing OOD scores...")
    ood_scores = compute_scores(model, processor, ood_loader, class_features, device)

    result = summarize(id_scores, ood_scores)

    print("\n========== Descriptor Results ==========")
    print(f"AUROC: {result['auroc'] * 100:.2f}")
    print(f"FPR95: {result['fpr95'] * 100:.2f}")
    print(f"ID mean: {result['id_mean']:.4f}")
    print(f"OOD mean: {result['ood_mean']:.4f}")
    print("=======================================")

    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()