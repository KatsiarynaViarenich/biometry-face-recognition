import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def compute_roc_eer(scores: np.ndarray, labels: np.ndarray):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        eer_threshold = float(interp1d(fpr, thresholds)(eer))
    except ValueError:
        eer = float('nan')
        eer_threshold = float('nan')

    return fpr, tpr, thresholds, roc_auc, eer, eer_threshold


def compute_tar_at_far(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray, target_far: float) -> tuple[float, float]:
    valid = np.where(fpr <= target_far)[0]
    if len(valid) == 0:
        return 0.0, float(thresholds[0])
    idx = valid[-1]
    return float(tpr[idx]), float(thresholds[idx])


def compute_threshold_metrics(
    true_pos_decisions: list[bool],
    true_neg_decisions: list[bool],
    fta_count: int,
    total_attempts: int,
) -> dict:
    n_pos = len(true_pos_decisions)
    n_neg = len(true_neg_decisions)

    false_rejects = sum(1 for ok in true_pos_decisions if not ok)
    false_accepts = sum(1 for ok in true_neg_decisions if ok)

    frr = false_rejects / n_pos if n_pos else 0.0
    far = false_accepts / n_neg if n_neg else 0.0
    fta = fta_count / total_attempts if total_attempts else 0.0
    tar = 1.0 - frr
    
    return {"far": far, "frr": frr, "fta": fta, "tar": tar,
            "false_rejects": false_rejects, "false_accepts": false_accepts,
            "fta_count": fta_count, "total_attempts": total_attempts}
