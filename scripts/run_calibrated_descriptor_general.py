import argparse
import json
import sys
from pathlib import Path
from collections import deque

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
    return [x[0] for x in batch], [int(x[1]) for x in batch]


def maybe_subsample(dataset, max_samples):
    if max_samples < 0:
        return dataset
    return Subset(dataset, list(range(min(max_samples, len(dataset)))))


def clean_descriptor(d):
    d = d.strip()
    d = d.lstrip("-•1234567890. ").strip()
    return d


def flatten_set(desc_set):
    return [clean_descriptor(d) for d in desc_set if clean_descriptor(d)]


@torch.no_grad()
def encode_texts(model, processor, texts, device):
    inputs = processor(text=texts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model.get_text_features(**inputs)
    feats = out.pooler_output if hasattr(out, "pooler_output") else out
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def largest_component(sim_matrix, eta):
    n = sim_matrix.shape[0]
    visited = [False] * n
    components = []

    for i in range(n):
        if visited[i]:
            continue

        q = deque([i])
        visited[i] = True
        comp = []

        while q:
            u = q.popleft()
            comp.append(u)

            for v in range(n):
                if not visited[v] and sim_matrix[u, v] >= eta:
                    visited[v] = True
                    q.append(v)

        components.append(comp)

    components.sort(key=len, reverse=True)
    return components[0]


@torch.no_grad()
def build_calibrated_text_features(
        model,
        processor,
        class_names,
        descriptors,
        device,
        eta=0.90,
        gamma=0.50,
):
    class_features = []
    calibration_info = {}

    for cls in tqdm(class_names, desc="Calibrating descriptors"):
        desc_sets = descriptors[cls]

        # Encode each descriptor set as one text feature
        set_texts = []
        for desc_set in desc_sets:
            descs = flatten_set(desc_set)
            text = f"a photo of a {cls} which has " + ", ".join(descs)
            set_texts.append(text)

        set_feats = encode_texts(model, processor, set_texts, device)
        sim = (set_feats @ set_feats.T).cpu().numpy()

        keep_indices = largest_component(sim, eta)
        p_c = len(keep_indices) / len(desc_sets)

        calibration_info[cls] = {
            "p_c": p_c,
            "keep_indices": keep_indices,
            "num_sets": len(desc_sets),
        }

        if p_c >= gamma:
            prompts = []
            for idx in keep_indices:
                descs = flatten_set(desc_sets[idx])
                prompts.append(f"a photo of a {cls}")
                for d in descs:
                    prompts.append(f"a photo of a {cls} which has {d}")
        else:
            prompts = [f"a photo of a {cls}"]

        feats = encode_texts(model, processor, prompts, device)
        class_feat = feats.mean(dim=0)
        class_feat = class_feat / class_feat.norm(dim=-1, keepdim=True)

        class_features.append(class_feat.cpu())

        print(f"{cls}: p(c)={p_c:.2f}, keep={keep_indices}")

    return torch.stack(class_features, dim=0), calibration_info


@torch.no_grad()
def compute_scores(model, processor, loader, text_features, device):
    all_scores = []
    text_features = text_features.to(device)

    for images, _ in tqdm(loader, desc="Computing calibrated descriptor scores"):
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
    parser.add_argument("--id_dataset", default="oxford_pet")
    parser.add_argument("--id_split", default="test")
    parser.add_argument("--ood_dataset", default="ninco")
    parser.add_argument("--ood_split", default="hard")
    parser.add_argument("--descriptor_path", default=str(PROJECT_ROOT / "descriptors" / "oxford_pet_gpt_descriptors_v2.json"))
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eta", type=float, default=0.90)
    parser.add_argument("--gamma", type=float, default=0.50)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "calibrated_descriptor_general_results.json"))
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)

    id_dataset_full = build_dataset(args.id_dataset, args.data_root, split=args.id_split)
    ood_dataset_full = build_dataset(args.ood_dataset, args.data_root, split=args.ood_split)

    class_names = get_class_names(id_dataset_full)

    id_dataset = maybe_subsample(id_dataset_full, args.max_samples)
    ood_dataset = maybe_subsample(ood_dataset_full, args.max_samples)

    descriptors = json.loads(Path(args.descriptor_path).read_text(encoding="utf-8"))

    id_loader = DataLoader(id_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_pil)
    ood_loader = DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_pil)

    text_features, calibration_info = build_calibrated_text_features(
        model=model,
        processor=processor,
        class_names=class_names,
        descriptors=descriptors,
        device=device,
        eta=args.eta,
        gamma=args.gamma,
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
        "eta": args.eta,
        "gamma": args.gamma,
        "calibration_info": calibration_info,
    })

    print("\n========== Calibrated Descriptor Results ==========")
    print(f"AUROC:   {result['auroc'] * 100:.2f}")
    print(f"FPR95:   {result['fpr95'] * 100:.2f}")
    print(f"ID mean: {result['id_mean']:.4f}")
    print(f"OOD mean:{result['ood_mean']:.4f}")
    print("=================================================")

    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()