import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.retrieval import build_retrieval_vectors, pairwise_retrieval_similarity


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate_pil(batch):
    return [x[0] for x in batch], [int(x[1]) for x in batch]


@torch.no_grad()
def encode_images(model, processor, loader, device):
    feats = []

    for images, _ in tqdm(loader, desc="Encoding image pool M"):
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        out = model.get_image_features(pixel_values=pixel_values)
        f = out.pooler_output if hasattr(out, "pooler_output") else out
        f = f / f.norm(dim=-1, keepdim=True)

        feats.append(f.cpu())

    return torch.cat(feats, dim=0)


@torch.no_grad()
def encode_descriptor_set(model, processor, class_name, desc_set, device):
    prompts = [f"a photo of a {class_name}"]

    for d in desc_set:
        prompts.append(f"a photo of a {class_name} which has {d}")

    inputs = processor(text=prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model.get_text_features(**inputs)
    feats = out.pooler_output if hasattr(out, "pooler_output") else out
    feats = feats / feats.norm(dim=-1, keepdim=True)

    return feats.cpu(), prompts


def main():
    model_path = PROJECT_ROOT / "models" / "clip-vit-base-patch16"
    descriptor_path = PROJECT_ROOT / "descriptors" / "cifar10_gpt_multi_descriptors.json"

    image_pool_size = 1000
    top_k = 50
    target_class = "dog"

    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading CLIP from {model_path}")
    model = CLIPModel.from_pretrained(str(model_path)).to(device)
    processor = CLIPProcessor.from_pretrained(str(model_path))

    print(f"Loading descriptors from {descriptor_path}")
    descriptors = json.loads(descriptor_path.read_text(encoding="utf-8"))

    print(f"Loading CIFAR10 train as unlabeled image pool M, size={image_pool_size}")
    dataset = CIFAR10(root=str(PROJECT_ROOT /"scripts" / "data"), train=True, download=True, transform=None)
    dataset = Subset(dataset, list(range(image_pool_size)))

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pil,
    )

    image_features = encode_images(model, processor, loader, device)

    print(f"\nAnalyzing class: {target_class}")
    desc_sets = descriptors[target_class]

    descriptor_features = []
    all_prompts = []

    for i, desc_set in enumerate(desc_sets):
        feats, prompts = encode_descriptor_set(model, processor, target_class, desc_set, device)
        descriptor_features.append(feats)
        all_prompts.append(prompts)

        print(f"\nDescriptor set {i + 1}:")
        for p in prompts:
            print("  ", p)

    retrieval_vectors = build_retrieval_vectors(
        image_features=image_features,
        descriptor_features=descriptor_features,
        top_k=top_k,
    )

    print("\nTop-k retrieved image indices:")
    for i, r in enumerate(retrieval_vectors):
        indices = torch.where(r)[0].tolist()
        print(f"Set {i + 1}: {indices[:20]} ... total={len(indices)}")

    sim = pairwise_retrieval_similarity(retrieval_vectors)

    print("\nPairwise retrieval cosine similarity:")
    for row in sim:
        print(["%.3f" % x for x in row])


if __name__ == "__main__":
    main()