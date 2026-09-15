import torch


@torch.no_grad()
def build_retrieval_vectors(
        image_features: torch.Tensor,
        descriptor_features: list[torch.Tensor],
        top_k: int = 50,
) -> list[torch.Tensor]:
    """
    image_features: (m, dim)
    descriptor_features: list，每个元素是一个 descriptor set 的 text features, shape=(num_prompts, dim)

    return:
        list of R(d), 每个 R(d) 是 bool tensor, shape=(m,)
    """
    retrieval_vectors = []

    for text_feats in descriptor_features:
        # 一个 descriptor set 里有多个文本 prompt，先和每张图片算相似度，再对 prompt 求平均
        sims = image_features @ text_feats.T          # (m, num_prompts)
        scores = sims.mean(dim=1)                     # (m,)

        top_indices = torch.topk(scores, k=top_k).indices

        r = torch.zeros(image_features.shape[0], dtype=torch.bool)
        r[top_indices.cpu()] = True
        retrieval_vectors.append(r)

    return retrieval_vectors


def retrieval_cosine(r1: torch.Tensor, r2: torch.Tensor) -> float:
    r1 = r1.float()
    r2 = r2.float()
    denom = r1.norm() * r2.norm()
    if denom.item() == 0:
        return 0.0
    return float((r1 @ r2 / denom).item())


def pairwise_retrieval_similarity(retrieval_vectors: list[torch.Tensor]) -> list[list[float]]:
    n = len(retrieval_vectors)
    sim = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            sim[i][j] = retrieval_cosine(retrieval_vectors[i], retrieval_vectors[j])

    return sim