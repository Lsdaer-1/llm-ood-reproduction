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


def collate_pil_with_index(batch):
    images = [x[0] for x in batch]
    labels = [int(x[1]) for x in batch]
    return images, labels


def maybe_subsample(dataset, max_samples):
    if max_samples < 0:
        return dataset
    return Subset(dataset, list(range(min(max_samples, len(dataset)))))


def clean_text(s):
    s = s.strip().lower()
    s = s.lstrip("-•1234567890. ").strip()
    s = s.replace(".", "").replace(",", "")
    return s


@torch.no_grad()
def encode_texts(model, processor, texts, device, batch_size=64):
    all_feats = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = model.get_text_features(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        feats = feats / feats.norm(dim=-1, keepdim=True)

        all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0)


@torch.no_grad()
def build_descriptor_features(model, processor, class_names, descriptors, device):
    features_by_class = []

    for cls in tqdm(class_names, desc="Encoding descriptor prompts"):
        desc_sets = descriptors[cls]

        prompts = []

        if len(desc_sets) == 0:
            prompts = [f"a photo of a {cls}"]
        else:
            for desc_set in desc_sets:
                for d in desc_set:
                    d = clean_text(d)
                    if not d:
                        continue
                    prompts.append(f"a photo of a {cls} which has {d}")

            if len(prompts) == 0:
                prompts = [f"a photo of a {cls}"]

        feats = encode_texts(model, processor, prompts, device)
        features_by_class.append(feats)

    return features_by_class


def load_object_cache(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cache = {}

    for k, v in raw.items():
        if isinstance(v, dict):
            objs = v.get("objects", [])
        elif isinstance(v, list):
            objs = v
        else:
            objs = []

        clean_objs = []
        for obj in objs:
            obj = clean_text(str(obj))
            if obj and obj != "unknown":
                clean_objs.append(obj)

        seen = set()
        dedup = []
        for obj in clean_objs:
            if obj not in seen:
                seen.add(obj)
                dedup.append(obj)

        cache[int(k)] = dedup

    return cache


@torch.no_grad()
def build_object_features_for_cache(model, processor, object_cache, device):
    all_objects = sorted({obj for objs in object_cache.values() for obj in objs})

    if len(all_objects) == 0:
        return {}, {}

    prompts = [f"a photo containing a {obj}" for obj in all_objects]
    feats = encode_texts(model, processor, prompts, device)

    obj_to_feat = {
        obj: feats[i]
        for i, obj in enumerate(all_objects)
    }

    return obj_to_feat, all_objects


@torch.no_grad()
def compute_scores(
        model,
        processor,
        loader,
        descriptor_features_by_class,
        object_cache,
        obj_to_feat,
        device,
        alpha,
):
    all_scores = []

    descriptor_features_by_class = [
        feats.to(device) for feats in descriptor_features_by_class
    ]

    obj_to_feat_device = {
        k: v.to(device) for k, v in obj_to_feat.items()
    }

    sample_offset = 0

    for images, _ in tqdm(loader, desc="Computing descriptor+object scores"):
        batch_size = len(images)

        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        out = model.get_image_features(pixel_values=pixel_values)
        image_features = out.pooler_output if hasattr(out, "pooler_output") else out
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        batch_class_scores = []

        for class_text_feats in descriptor_features_by_class:
            desc_sims = image_features @ class_text_feats.T
            desc_score = desc_sims.mean(dim=1)

            obj_scores = []

            for b in range(batch_size):
                sample_idx = sample_offset + b
                objs = object_cache.get(sample_idx, [])

                if len(objs) == 0:
                    obj_scores.append(torch.tensor(0.0, device=device))
                    continue

                obj_feats = [
                    obj_to_feat_device[o]
                    for o in objs
                    if o in obj_to_feat_device
                ]

                if len(obj_feats) == 0:
                    obj_scores.append(torch.tensor(0.0, device=device))
                    continue

                obj_feats = torch.stack(obj_feats, dim=0)
                obj_sims = obj_feats @ class_text_feats.T
                obj_score = obj_sims.max()

                obj_scores.append(obj_score)

            obj_score = torch.stack(obj_scores)

            final_score = alpha * desc_score + (1.0 - alpha) * obj_score
            batch_class_scores.append(final_score)

        batch_class_scores = torch.stack(batch_class_scores, dim=1)
        scores = batch_class_scores.max(dim=1).values

        all_scores.append(scores.cpu().numpy())
        sample_offset += batch_size

    return np.concatenate(all_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=str(PROJECT_ROOT / "models" / "clip-vit-base-patch16"))
    parser.add_argument("--data_root", default=str(PROJECT_ROOT / "data"))

    parser.add_argument("--id_dataset", default="oxford_pet")
    parser.add_argument("--id_split", default="test")
    parser.add_argument("--ood_dataset", default="ninco")
    parser.add_argument("--ood_split", default="hard")

    parser.add_argument("--descriptor_path", default=str(PROJECT_ROOT / "descriptors" / "oxford_pet_gpt_descriptors_retrieval_calibrated_eta090.json"))
    parser.add_argument("--id_object_path", default=str(PROJECT_ROOT / "results" / "gpt_objects_oxford_pet_500.json"))
    parser.add_argument("--ood_object_path", default=str(PROJECT_ROOT / "results" / "gpt_objects_ninco_hard_500.json"))

    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.5)

    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "descriptor_object_general_results.json"))
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading CLIP: {args.model_name}")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model.eval()

    print("Loading datasets...")
    id_dataset_full = build_dataset(args.id_dataset, args.data_root, split=args.id_split)
    ood_dataset_full = build_dataset(args.ood_dataset, args.data_root, split=args.ood_split)

    class_names = get_class_names(id_dataset_full)

    id_dataset = maybe_subsample(id_dataset_full, args.max_samples)
    ood_dataset = maybe_subsample(ood_dataset_full, args.max_samples)

    print(f"ID samples: {len(id_dataset)}")
    print(f"OOD samples: {len(ood_dataset)}")
    print(f"Classes: {len(class_names)}")

    print(f"Loading descriptors: {args.descriptor_path}")
    descriptors = json.loads(Path(args.descriptor_path).read_text(encoding="utf-8"))

    print(f"Loading ID object cache: {args.id_object_path}")
    id_object_cache = load_object_cache(args.id_object_path)

    print(f"Loading OOD object cache: {args.ood_object_path}")
    ood_object_cache = load_object_cache(args.ood_object_path)

    print("Encoding descriptor features...")
    descriptor_features_by_class = build_descriptor_features(
        model=model,
        processor=processor,
        class_names=class_names,
        descriptors=descriptors,
        device=device,
    )

    print("Encoding object features...")
    merged_object_cache = {}
    merged_object_cache.update(id_object_cache)
    offset = 10_000_000
    for k, v in ood_object_cache.items():
        merged_object_cache[k + offset] = v

    obj_to_feat, all_objects = build_object_features_for_cache(
        model=model,
        processor=processor,
        object_cache=merged_object_cache,
        device=device,
    )

    print(f"Unique object words: {len(all_objects)}")
    print("First 30 objects:", all_objects[:30])

    id_loader = DataLoader(
        id_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pil_with_index,
    )

    ood_loader = DataLoader(
        ood_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pil_with_index,
    )

    print("Computing ID scores...")
    id_scores = compute_scores(
        model=model,
        processor=processor,
        loader=id_loader,
        descriptor_features_by_class=descriptor_features_by_class,
        object_cache=id_object_cache,
        obj_to_feat=obj_to_feat,
        device=device,
        alpha=args.alpha,
    )

    print("Computing OOD scores...")
    ood_scores = compute_scores(
        model=model,
        processor=processor,
        loader=ood_loader,
        descriptor_features_by_class=descriptor_features_by_class,
        object_cache=ood_object_cache,
        obj_to_feat=obj_to_feat,
        device=device,
        alpha=args.alpha,
    )

    result = summarize(id_scores, ood_scores)
    result.update({
        "id_dataset": args.id_dataset,
        "ood_dataset": args.ood_dataset,
        "ood_split": args.ood_split,
        "id_samples": len(id_scores),
        "ood_samples": len(ood_scores),
        "num_classes": len(class_names),
        "alpha": args.alpha,
        "descriptor_path": args.descriptor_path,
        "id_object_path": args.id_object_path,
        "ood_object_path": args.ood_object_path,
        "num_object_words": len(all_objects),
    })

    print("\n========== Descriptor + Object Results ==========")
    print(f"Alpha:    {args.alpha}")
    print(f"AUROC:    {result['auroc'] * 100:.2f}")
    print(f"FPR95:    {result['fpr95'] * 100:.2f}")
    print(f"ID mean:  {result['id_mean']:.4f}")
    print(f"OOD mean: {result['ood_mean']:.4f}")
    print("================================================")

    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()