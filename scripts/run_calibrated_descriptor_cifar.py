import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.metrics import summarize
from utils.retrieval import build_retrieval_vectors
from utils.calibration import calibrate_descriptor_sets


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


def collate_pil(batch):
    return [x[0] for x in batch], [int(x[1]) for x in batch]


def build_dataset(name, root, train=False):
    if name == "cifar10":
        return CIFAR10(root=root, train=train, download=True, transform=None)
    if name == "cifar100":
        return CIFAR100(root=root, train=False, download=True, transform=None)
    if name == "svhn":
        return SVHN(root=root, split="test", download=True, transform=None)
    raise ValueError(name)


def maybe_subsample(dataset, max_samples):
    if max_samples < 0:
        return dataset
    return Subset(dataset, list(range(min(max_samples, len(dataset)))))


@torch.no_grad()
def encode_images(model, processor, loader, device):
    feats = []

    for images, _ in tqdm(loader, desc="Encoding images"):
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        out = model.get_image_features(pixel_values=pixel_values)
        f = out.pooler_output if hasattr(out, "pooler_output") else out
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())

    return torch.cat(feats, dim=0)


@torch.no_grad()
def encode_prompt_set(
        model,
        processor,
        class_name,
        desc_set,
        device,
        include_class_prompt=True,
):
    prompts = []

    if include_class_prompt:
        prompts.append(f"a photo of a {class_name}")

    for d in desc_set:
        d = d.strip()
        if d and not d.lower().startswith("there are several useful visual features"):
            prompts.append(f"a photo of a {class_name} which has {d}")

    # 防止 descriptor 全空
    if len(prompts) == 0:
        prompts.append(f"a photo of a {class_name}")

    inputs = processor(text=prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model.get_text_features(**inputs)
    feats = out.pooler_output if hasattr(out, "pooler_output") else out
    feats = feats / feats.norm(dim=-1, keepdim=True)

    return feats.cpu()


@torch.no_grad()
def encode_calibrated_class_features(
        model,
        processor,
        descriptors,
        image_pool_features,
        device,
        top_k=50,
        eta=0.9,
        gamma=0.5,
):
    class_features = []
    calibration_info = {}

    for cls in CIFAR10_CLASSES:
        desc_sets = descriptors[cls]

        descriptor_features = []
        for desc_set in desc_sets:
            feats = encode_prompt_set(model, processor, cls, desc_set, device, include_class_prompt=False,)
            descriptor_features.append(feats)

        retrieval_vectors = build_retrieval_vectors(
            image_features=image_pool_features,
            descriptor_features=descriptor_features,
            top_k=top_k,
        )

        keep_indices, confidence, sim_matrix = calibrate_descriptor_sets(
            retrieval_vectors,
            eta=eta,
        )

        calibration_info[cls] = {
            "confidence": confidence,
            "keep_indices": keep_indices,
            "num_sets": len(desc_sets),
        }

        print(f"{cls}: p(c)={confidence:.2f}, keep={keep_indices}")

        if confidence >= gamma:
            kept_feats = [descriptor_features[i] for i in keep_indices]
            merged = torch.cat(kept_feats, dim=0)
        else:
            merged = encode_prompt_set(model, processor, cls, [], device)

        class_features.append(merged)

    return class_features, calibration_info


@torch.no_grad()
def compute_scores(model, processor, loader, class_features, device):
    all_scores = []
    model.eval()

    for images, _ in tqdm(loader, desc="Computing calibrated descriptor scores"):
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        out = model.get_image_features(pixel_values=pixel_values)
        image_features = out.pooler_output if hasattr(out, "pooler_output") else out
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        class_scores = []

        for feats in class_features:
            feats = feats.to(device)
            sims = image_features @ feats.T
            score = sims.mean(dim=1)
            class_scores.append(score)

        class_scores = torch.stack(class_scores, dim=1)
        final_scores = class_scores.max(dim=1).values
        all_scores.append(final_scores.cpu().numpy())

    return np.concatenate(all_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=str(PROJECT_ROOT / "models" / "clip-vit-base-patch16"))
    parser.add_argument("--descriptor_path", default=str(PROJECT_ROOT / "descriptors" / "cifar10_gpt4.1_multi_descriptors.json"))
    parser.add_argument("--data_root", default=str(PROJECT_ROOT / "scripts" / "data"))
    parser.add_argument("--ood_dataset", default="cifar100", choices=["cifar100", "svhn"])
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--pool_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "calibrated_descriptor_cifar_results.json"))
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading CLIP: {args.model_name}")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)

    print(f"Loading descriptors: {args.descriptor_path}")
    descriptors = json.loads(Path(args.descriptor_path).read_text(encoding="utf-8"))

    print("Building unlabeled image pool M from CIFAR10 train...")
    pool_dataset = build_dataset("cifar10", args.data_root, train=True)
    pool_dataset = maybe_subsample(pool_dataset, args.pool_size)
    pool_loader = DataLoader(pool_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pil)

    image_pool_features = encode_images(model, processor, pool_loader, device)

    print("\nCalibrating descriptor sets...")
    class_features, calibration_info = encode_calibrated_class_features(
        model=model,
        processor=processor,
        descriptors=descriptors,
        image_pool_features=image_pool_features,
        device=device,
        top_k=args.top_k,
        eta=args.eta,
        gamma=args.gamma,
    )

    print("\nLoading ID/OOD datasets...")
    id_dataset = maybe_subsample(build_dataset("cifar10", args.data_root, train=False), args.max_samples)
    ood_dataset = maybe_subsample(build_dataset(args.ood_dataset, args.data_root), args.max_samples)

    id_loader = DataLoader(id_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pil)
    ood_loader = DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pil)

    print("Computing ID scores...")
    id_scores = compute_scores(model, processor, id_loader, class_features, device)

    print("Computing OOD scores...")
    ood_scores = compute_scores(model, processor, ood_loader, class_features, device)

    result = summarize(id_scores, ood_scores)
    result["calibration"] = calibration_info

    print("\n========== Calibrated Descriptor Results ==========")
    print(f"AUROC: {result['auroc'] * 100:.2f}")
    print(f"FPR95: {result['fpr95'] * 100:.2f}")
    print(f"ID mean: {result['id_mean']:.4f}")
    print(f"OOD mean: {result['ood_mean']:.4f}")
    print("=================================================")

    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()