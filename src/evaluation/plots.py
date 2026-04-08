import os
import numpy as np
import matplotlib.pyplot as plt


def _make_filename(prefix: str, title_suffix: str) -> str:
    safe = title_suffix.replace(' ', '_') if title_suffix else ""
    return f"{prefix}_{safe}.png" if safe else f"{prefix}.png"


def plot_roc(fpr, tpr, roc_auc: float, eer: float, out_dir: str, title_suffix: str = ""):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    label = f'ROC (AUC = {roc_auc:.4f}'
    if not np.isnan(eer):
        label += f', EER = {eer:.4f})'
    else:
        label += ')'
    plt.plot(fpr, tpr, color='blue', lw=2, label=label)
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (TAR)')
    plt.title(f'ROC {title_suffix}')
    plt.legend(loc="lower right")
    plt.grid(True)

    fname = _make_filename("roc_curve", title_suffix)
    path = os.path.join(out_dir, fname)
    plt.savefig(path)
    plt.close()
    return path


def plot_far_frr_vs_threshold(
    fpr, tpr, thresholds, eer: float, eer_threshold: float,
    out_dir: str, title_suffix: str = ""
):
    os.makedirs(out_dir, exist_ok=True)

    fnr = 1.0 - tpr
    valid = np.where((thresholds >= 0) & (thresholds <= 1.0))[0]
    thresh_plot = thresholds[valid]
    far_plot = fpr[valid]
    frr_plot = fnr[valid]

    plt.figure(figsize=(8, 6))
    plt.plot(thresh_plot, far_plot, label='FAR', color='red')
    plt.plot(thresh_plot, frr_plot, label='FRR', color='blue')
    if not np.isnan(eer_threshold):
        plt.axvline(
            x=eer_threshold, color='black', linestyle='--',
            label=f'EER Threshold ≈ {eer_threshold:.3f} (EER = {eer*100:.2f}%)'
        )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Decision Threshold')
    plt.ylabel('Error Rate')
    plt.title(f'FAR and FRR vs Threshold {title_suffix}')
    plt.legend()
    plt.grid(True)

    fname = _make_filename("far_frr_curve", title_suffix)
    path = os.path.join(out_dir, fname)
    plt.savefig(path)
    plt.close()
    return path
