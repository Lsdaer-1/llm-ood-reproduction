import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    y_true = np.concatenate([
        np.ones_like(id_scores, dtype=np.int64),
        np.zeros_like(ood_scores, dtype=np.int64),
    ])
    y_score = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(y_true, y_score))


def compute_fpr95(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    threshold = np.percentile(id_scores, 5)
    return float(np.mean(ood_scores >= threshold))


def summarize(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict:
    return {
        "auroc": compute_auroc(id_scores, ood_scores),
        "fpr95": compute_fpr95(id_scores, ood_scores),
        "id_mean": float(np.mean(id_scores)),
        "ood_mean": float(np.mean(ood_scores)),
    }
