import torch


COCO_OBJECTS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


@torch.no_grad()
def encode_concept_texts(model, processor, device, concepts=None):
    if concepts is None:
        concepts = COCO_OBJECTS

    prompts = [f"a photo of a {c}" for c in concepts]
    inputs = processor(text=prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model.get_text_features(**inputs)
    feats = out.pooler_output if hasattr(out, "pooler_output") else out
    feats = feats / feats.norm(dim=-1, keepdim=True)

    return concepts, feats


@torch.no_grad()
def detect_concepts_from_image_features(
        image_features,
        concept_features,
        concepts,
        top_k=3,
):
    """
    image_features: (B, dim)
    concept_features: (num_concepts, dim)

    return:
        list[list[str]], 每张图 top-k concepts
    """
    sims = image_features @ concept_features.T
    top_indices = torch.topk(sims, k=top_k, dim=1).indices

    batch_concepts = []
    for row in top_indices:
        batch_concepts.append([concepts[int(i)] for i in row])

    return batch_concepts