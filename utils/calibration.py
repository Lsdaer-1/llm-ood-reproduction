import torch
from collections import deque


def largest_connected_component(sim_matrix, eta=0.9):
    n = len(sim_matrix)
    visited = [False] * n
    best = []

    for start in range(n):
        if visited[start]:
            continue

        queue = deque([start])
        visited[start] = True
        comp = []

        while queue:
            i = queue.popleft()
            comp.append(i)

            for j in range(n):
                if not visited[j] and sim_matrix[i][j] >= eta:
                    visited[j] = True
                    queue.append(j)

        if len(comp) > len(best):
            best = comp

    return sorted(best)


def calibrate_descriptor_sets(retrieval_vectors, eta=0.9):
    n = len(retrieval_vectors)
    sim = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            r1 = retrieval_vectors[i].float()
            r2 = retrieval_vectors[j].float()
            denom = r1.norm() * r2.norm()
            sim[i][j] = 0.0 if denom.item() == 0 else float((r1 @ r2 / denom).item())

    keep_indices = largest_connected_component(sim, eta=eta)
    confidence = len(keep_indices) / n

    return keep_indices, confidence, sim