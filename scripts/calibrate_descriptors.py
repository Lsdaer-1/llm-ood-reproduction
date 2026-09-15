import argparse
import json
import sys
from pathlib import Path
from collections import deque

import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset_builder import build_dataset, get_class_names


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clean_descriptor(d: str) -> str:
    d = d.strip()
    d = d.lstrip("-•1234567890. ").strip()
    return d


def flatten_set(desc_set):
    return [clean_descriptor(d) for d in desc_set if clean_descriptor(d)]


@torch.no_grad()
def encode_texts(model, processor, texts, device, batch_size=64):
    feats_all = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        inputs = processor(text=batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = model.get_text_features(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        feats = feats / feats.norm(dim=-1, keepdim=True)

        feats_all.append(feats.cpu())

    return torch.cat(feats_all, dim=0)


def retrieve_topk(text_feats, image_feats, topk):
    sims = text_feats @ image_feats.T
    _, indices = torch.topk(sims, k=topk, dim=1)
    return indices.cpu()


def retrieval_overlap_matrix(topk_indices):
    n = topk_indices.shape[0]
    sim = torch.zeros((n, n), dtype=torch.float32)

    sets = [set(topk_indices[i].tolist()) for i in range(n)]

    for i in range(n):
        for j in range(n):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            sim[i, j] = inter / union if union > 0 else 0.0

    return sim


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=str(PROJECT_ROOT / "models" / "clip-vit-base-patch16"))
    parser.add_argument("--data_root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--id_dataset", default="oxford_pet")
    parser.add_argument("--id_split", default="test")
    parser.add_argument("--descriptor_path", default=str(PROJECT_ROOT / "descriptors" / "oxford_pet_gpt_descriptors_v2.json"))
    parser.add_argument("--retrieval_db", default=str(PROJECT_ROOT / "results" / "tiny_imagenet_retrieval_db.pt"))
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.30)
    parser.add_argument("--gamma", type=float, default=0.50)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "descriptors" / "oxford_pet_gpt_descriptors_retrieval_calibrated.json"))
    parser.add_argument("--info_output", default=str(PROJECT_ROOT / "results" / "retrieval_calibration_info.json"))
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading CLIP: {args.model_name}")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model.eval()

    print("Loading ID dataset...")
    id_dataset = build_dataset(args.id_dataset, args.data_root, split=args.id_split)
    class_names = get_class_names(id_dataset)

    print(f"Loading descriptors: {args.descriptor_path}")
    descriptors = json.loads(Path(args.descriptor_path).read_text(encoding="utf-8"))

    print(f"Loading retrieval DB: {args.retrieval_db}")
    db = torch.load(args.retrieval_db, map_location="cpu")
    image_feats = db["features"]
    image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)

    calibrated = {}
    info = {}

    for cls in tqdm(class_names, desc="Retrieval calibration"):
        desc_sets = descriptors[cls]

        set_prompts = []
        for desc_set in desc_sets:
            descs = flatten_set(desc_set)
            prompt = f"a photo of a {cls} which has " + ", ".join(descs)
            set_prompts.append(prompt)

        text_feats = encode_texts(model, processor, set_prompts, device)

        topk_indices = retrieve_topk(
            text_feats=text_feats,
            image_feats=image_feats,
            topk=args.topk,
        )

        overlap = retrieval_overlap_matrix(topk_indices)
        keep = largest_component(overlap, args.eta)
        p_c = len(keep) / len(desc_sets)

        if p_c >= args.gamma:
            calibrated[cls] = [desc_sets[i] for i in keep]
        else:
            calibrated[cls] = []

        info[cls] = {
            "p_c": p_c,
            "keep_indices": keep,
            "num_sets": len(desc_sets),
            "used_descriptors": p_c >= args.gamma,
            "eta": args.eta,
            "gamma": args.gamma,
            "topk": args.topk,
        }

        print(
            f"{cls:30s} p(c)={p_c:.2f} "
            f"keep={keep} "
            f"used={p_c >= args.gamma}"
        )

    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(calibrated, indent=2, ensure_ascii=False), encoding="utf-8")

    info_out = Path(args.info_output)
    info_out.parent.mkdir(exist_ok=True)
    info_out.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved calibrated descriptors to: {out}")
    print(f"Saved calibration info to: {info_out}")


if __name__ == "__main__":
    main()