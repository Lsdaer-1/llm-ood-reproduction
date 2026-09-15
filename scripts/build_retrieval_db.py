import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset_builder import build_dataset


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


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=str(PROJECT_ROOT / "models" / "clip-vit-base-patch16"))
    parser.add_argument("--data_root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--dataset", default="tiny_imagenet")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=50000)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "tiny_imagenet_retrieval_db.pt"))
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading CLIP: {args.model_name}")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model.eval()

    print(f"Loading retrieval dataset: {args.dataset}, split={args.split}")
    dataset = build_dataset(args.dataset, args.data_root, split=args.split)

    if args.max_samples > 0:
        n = min(args.max_samples, len(dataset))
        dataset = Subset(dataset, list(range(n)))
    else:
        n = len(dataset)

    print(f"Retrieval samples: {n}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pil_with_index,
    )

    all_features = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Encoding retrieval images"):
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        out = model.get_image_features(pixel_values=pixel_values)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        feats = feats / feats.norm(dim=-1, keepdim=True)

        all_features.append(feats.cpu())
        all_labels.extend(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.tensor(all_labels, dtype=torch.long)

    result = {
        "features": features,
        "labels": labels,
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": int(features.shape[0]),
        "model_name": args.model_name,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    torch.save(result, out_path)

    print(f"Saved retrieval DB to: {out_path}")
    print(f"features shape: {features.shape}")
    print(f"labels shape: {labels.shape}")


if __name__ == "__main__":
    main()