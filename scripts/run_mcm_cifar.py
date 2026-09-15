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


def get_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def parse_args():
    parser = argparse.ArgumentParser(description="MCM baseline on CIFAR10 as ID.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="../models/clip-vit-base-patch16",
    )
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--id_dataset", type=str, default="cifar10", choices=["cifar10"])
    parser.add_argument("--ood_dataset", type=str, default="cifar100", choices=["cifar100", "svhn"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--output", type=str, default="./results/mcm_cifar_results.json")
    return parser.parse_args()


def build_dataset(name: str, root: str):
    root = Path(root)
    if name == "cifar10":
        return CIFAR10(root=root, train=False, download=True, transform=None)
    if name == "cifar100":
        return CIFAR100(root=root, train=False, download=True, transform=None)
    if name == "svhn":
        return SVHN(root=root, split="test", download=True, transform=None)
    raise ValueError(f"Unknown dataset: {name}")


def maybe_subsample(dataset, max_samples: int):
    if max_samples is None or max_samples < 0:
        return dataset
    n = min(max_samples, len(dataset))
    return Subset(dataset, list(range(n)))


def collate_pil(batch) -> Tuple[List[Image.Image], List[int]]:
    images = [x[0] for x in batch]
    labels = [int(x[1]) for x in batch]
    return images, labels


@torch.no_grad()
def encode_text_features(model: CLIPModel, processor: CLIPProcessor,
                         class_names: List[str], device: torch.device) -> torch.Tensor:
    prompts = [f"a photo of a {name}" for name in class_names]
    inputs = processor(text=prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.get_text_features(**inputs)
    text_features = out.pooler_output if hasattr(out, "pooler_output") else out
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


@torch.no_grad()
def compute_mcm_scores(model: CLIPModel, processor: CLIPProcessor,
                       loader: DataLoader, text_features: torch.Tensor,
                       device: torch.device) -> np.ndarray:
    all_scores = []
    model.eval()
    for images, _ in tqdm(loader, desc="Computing MCM scores"):
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        out = model.get_image_features(pixel_values=pixel_values)
        image_features = out.pooler_output if hasattr(out, "pooler_output") else out
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ text_features.T
        scores = logits.max(dim=1).values
        all_scores.append(scores.detach().cpu().numpy())
    return np.concatenate(all_scores, axis=0)


def main():
    args = parse_args()
    device = get_device(args.device)

    print(f"Using device: {device}")
    print(f"Loading CLIP model: {args.model_name}")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)

    print("Loading datasets...")
    id_dataset = maybe_subsample(build_dataset(args.id_dataset, args.data_root), args.max_samples)
    ood_dataset = maybe_subsample(build_dataset(args.ood_dataset, args.data_root), args.max_samples)

    id_loader = DataLoader(id_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, collate_fn=collate_pil)
    ood_loader = DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_pil)

    print("Encoding CIFAR10 class prompts...")
    text_features = encode_text_features(model, processor, CIFAR10_CLASSES, device)

    print("Computing ID scores...")
    id_scores = compute_mcm_scores(model, processor, id_loader, text_features, device)

    print("Computing OOD scores...")
    ood_scores = compute_mcm_scores(model, processor, ood_loader, text_features, device)

    result = summarize(id_scores, ood_scores)
    result.update({
        "id_dataset": args.id_dataset,
        "ood_dataset": args.ood_dataset,
        "model_name": args.model_name,
        "device": str(device),
        "id_samples": int(len(id_scores)),
        "ood_samples": int(len(ood_scores)),
        "classes": CIFAR10_CLASSES,
    })

    print("\\n========== MCM Results ==========")
    print(f"ID dataset:   {args.id_dataset}")
    print(f"OOD dataset:  {args.ood_dataset}")
    print(f"ID samples:   {len(id_scores)}")
    print(f"OOD samples:  {len(ood_scores)}")
    print(f"AUROC:        {result['auroc'] * 100:.2f}")
    print(f"FPR95:        {result['fpr95'] * 100:.2f}")
    print(f"ID mean:      {result['id_mean']:.4f}")
    print(f"OOD mean:     {result['ood_mean']:.4f}")
    print("=================================")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved result to: {output_path}")


if __name__ == "__main__":
    main()
