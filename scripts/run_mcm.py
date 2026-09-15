import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.metrics import compute_auroc, compute_fpr95


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def get_auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--id_dataset", type=str, default="cifar10", choices=["cifar10"])
    parser.add_argument("--ood_dataset", type=str, default="cifar100", choices=["cifar100", "svhn"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="ViT-B-16")
    parser.add_argument("--max_samples", type=int, default=-1)
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


def collate_pil(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return images, labels


def maybe_subsample(dataset, max_samples: int):
    if max_samples is None or max_samples < 0:
        return dataset
    indices = list(range(min(max_samples, len(dataset))))
    return torch.utils.data.Subset(dataset, indices)


def encode_text_features(
    model: CLIPModel,
    processor: CLIPProcessor,
    class_names: List[str],
    device: torch.device,
) -> torch.Tensor:
    prompts = [f"a photo of a {name}" for name in class_names]

    inputs = processor(
        text=prompts,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features


def compute_mcm_scores(
    model: CLIPModel,
    processor: CLIPProcessor,
    loader: DataLoader,
    text_features: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    all_scores = []

    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Computing scores"):
            inputs = processor(
                images=images,
                return_tensors="pt",
            )
            pixel_values = inputs["pixel_values"].to(device)

            image_features = model.get_image_features(pixel_values=pixel_values)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = image_features @ text_features.T
            scores = logits.max(dim=1).values

            all_scores.append(scores.detach().cpu().numpy())

    return np.concatenate(all_scores, axis=0)


def main():
    args = parse_args()

    if args.device == "auto":
        device = get_auto_device()
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，切换到 CPU。")
        device = torch.device("cpu")

    if device.type == "mps" and not torch.backends.mps.is_available():
        print("MPS 不可用，切换到 CPU。")
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Loading CLIP model: {args.model_name}")

    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)

    print("Loading datasets...")
    id_dataset = maybe_subsample(build_dataset(args.id_dataset, args.data_root), args.max_samples)
    ood_dataset = maybe_subsample(build_dataset(args.ood_dataset, args.data_root), args.max_samples)

    id_loader = DataLoader(
        id_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pil,
    )
    ood_loader = DataLoader(
        ood_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pil,
    )

    print("Encoding ID class names...")
    text_features = encode_text_features(model, processor, CIFAR10_CLASSES, device)

    print("Computing ID scores...")
    id_scores = compute_mcm_scores(model, processor, id_loader, text_features, device)

    print("Computing OOD scores...")
    ood_scores = compute_mcm_scores(model, processor, ood_loader, text_features, device)

    auroc = compute_auroc(id_scores, ood_scores)
    fpr95 = compute_fpr95(id_scores, ood_scores)

    print("\n========== Results ==========")
    print(f"ID dataset:  {args.id_dataset}")
    print(f"OOD dataset: {args.ood_dataset}")
    print(f"Model:       {args.model_name}")
    print(f"Device:      {device}")
    print(f"ID samples:  {len(id_scores)}")
    print(f"OOD samples: {len(ood_scores)}")
    print(f"AUROC:       {auroc * 100:.2f}")
    print(f"FPR95:       {fpr95 * 100:.2f}")
    print("=============================")


if __name__ == "__main__":
    main()
