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
from utils.concept_detector import encode_concept_texts, detect_concepts_from_image_features


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


def load_descriptors(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cleaned = {}

    for cls, desc_sets in data.items():
        # 兼容 single descriptors 和 multi descriptors
        if len(desc_sets) > 0 and isinstance(desc_sets[0], str):
            desc_sets = [desc_sets]

        merged = []
        for desc_set in desc_sets:
            for d in desc_set:
                d = d.strip()
                if not d:
                    continue
                if d.lower().startswith("there are several useful visual features"):
                    continue
                merged.append(d)

        cleaned[cls] = merged

    return cleaned


@torch.no_grad()
def encode_class_text_features(model, processor, descriptors, device):
    class_features = {}

    for cls in CIFAR10_CLASSES:
        prompts = [f"a photo of a {cls}"]

        for d in descriptors.get(cls, []):
            prompts.append(f"a photo of a {cls} which has {d}")

        inputs = processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = model.get_text_features(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        feats = feats / feats.norm(dim=-1, keepdim=True)

        class_features[cls] = feats

    return class_features


@torch.no_grad()
def compute_scores_with_concepts(
        model,
        processor,
        loader,
        class_features,
        concept_names,
        concept_features,
        device,
        concept_top_k=3,
        alpha=1.0,
):
    all_scores = []
    model.eval()

    for images, _ in tqdm(loader, desc="Computing concept descriptor scores"):
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        out = model.get_image_features(pixel_values=pixel_values)
        image_features = out.pooler_output if hasattr(out, "pooler_output") else out
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        batch_concepts = detect_concepts_from_image_features(
            image_features=image_features,
            concept_features=concept_features,
            concepts=concept_names,
            top_k=concept_top_k,
        )

        class_scores = []

        for cls in CIFAR10_CLASSES:
            text_feats = class_features[cls].to(device)

            # 公式左半项：E_t sigma(I(x), T(t))
            image_text_sims = image_features @ text_feats.T
            left_score = image_text_sims.mean(dim=1)

            # 公式右半项：E_v E_t sigma(T(v), T(t))
            right_scores = []

            for concepts_for_one_image in batch_concepts:
                prompts = [f"a photo of a {v}" for v in concepts_for_one_image]
                concept_inputs = processor(text=prompts, return_tensors="pt", padding=True)
                concept_inputs = {k: v.to(device) for k, v in concept_inputs.items()}

                out_v = model.get_text_features(**concept_inputs)
                v_feats = out_v.pooler_output if hasattr(out_v, "pooler_output") else out_v
                v_feats = v_feats / v_feats.norm(dim=-1, keepdim=True)

                sim_vt = v_feats @ text_feats.T
                right_scores.append(sim_vt.mean())

            right_score = torch.stack(right_scores).to(device)

            score = left_score + alpha * right_score
            class_scores.append(score)

        class_scores = torch.stack(class_scores, dim=1)
        final_scores = class_scores.max(dim=1).values

        all_scores.append(final_scores.cpu().numpy())

    return np.concatenate(all_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=str(PROJECT_ROOT / "models" / "clip-vit-base-patch16"))
    parser.add_argument("--descriptor_path", default=str(PROJECT_ROOT / "descriptors" / "cifar10_gpt4.1_multi_descriptors.json"))
    parser.add_argument("--data_root", default=str(PROJECT_ROOT /"scripts"/ "data"))
    parser.add_argument("--ood_dataset", default="cifar100", choices=["cifar100", "svhn"])
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--concept_top_k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "concept_descriptor_cifar_results.json"))
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading CLIP: {args.model_name}")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)

    print(f"Loading descriptors: {args.descriptor_path}")
    descriptors = load_descriptors(args.descriptor_path)

    print("Encoding class descriptor prompts...")
    class_features = encode_class_text_features(model, processor, descriptors, device)

    print("Encoding concept vocabulary...")
    concept_names, concept_features = encode_concept_texts(model, processor, device)

    print("Loading ID/OOD datasets...")
    id_dataset = maybe_subsample(build_dataset("cifar10", args.data_root, train=False), args.max_samples)
    ood_dataset = maybe_subsample(build_dataset(args.ood_dataset, args.data_root), args.max_samples)

    id_loader = DataLoader(id_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pil)
    ood_loader = DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pil)

    print("Computing ID scores...")
    id_scores = compute_scores_with_concepts(
        model, processor, id_loader, class_features,
        concept_names, concept_features, device,
        concept_top_k=args.concept_top_k,
        alpha=args.alpha,
    )

    print("Computing OOD scores...")
    ood_scores = compute_scores_with_concepts(
        model, processor, ood_loader, class_features,
        concept_names, concept_features, device,
        concept_top_k=args.concept_top_k,
        alpha=args.alpha,
    )

    result = summarize(id_scores, ood_scores)
    result.update({
        "concept_top_k": args.concept_top_k,
        "alpha": args.alpha,
    })

    print("\n========== Concept Descriptor Results ==========")
    print(f"AUROC: {result['auroc'] * 100:.2f}")
    print(f"FPR95: {result['fpr95'] * 100:.2f}")
    print(f"ID mean: {result['id_mean']:.4f}")
    print(f"OOD mean: {result['ood_mean']:.4f}")
    print("==============================================")

    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()