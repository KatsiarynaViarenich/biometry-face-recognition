import os
import numpy as np
from tqdm import tqdm

from evaluation.metrics import compute_roc_eer, compute_tar_at_far, compute_threshold_metrics
from evaluation.plots import plot_roc, plot_far_frr_vs_threshold

PLOTS_DIR = "experiments/plots"


def _resolve(img_or_path, data_root: str):
    if isinstance(img_or_path, str):
        # Jeśli ścieżka już zaczyna się od 'data/', nie doklejamy DATA_ROOT
        if img_or_path.startswith("data/"):
            return img_or_path
        if data_root and os.path.isabs(img_or_path):
             return img_or_path
        if data_root and img_or_path.startswith(data_root):
             return img_or_path
        return os.path.join(data_root, img_or_path) if data_root else img_or_path
    return img_or_path


def _load_img(img_or_path):
    from PIL import Image
    if isinstance(img_or_path, Image.Image):
        return img_or_path.convert('RGB')
    return Image.open(img_or_path).convert('RGB')


def collect_scores(system, test_pos, test_neg, data_root: str, threshold: float, pos_limit: int, neg_limit: int, desc: str = "", priority_ids=None):
    scores, labels = [], []
    pos_decisions, neg_decisions = [], []
    false_rejects, false_accepts = [], []
    priority_results = []
    
    fta_count = 0
    total_attempts = 0
    
    p_ids = set(priority_ids) if priority_ids else set()
    random_demo_count = 0
    RANDOM_DEMO_LIMIT = 10

    for img_or_path, true_uid in tqdm(test_pos[:pos_limit], desc=f"Genuine {desc}"):
        total_attempts += 1
        resolved_path = _resolve(img_or_path, data_root)
        allowed, score = system.authenticate(resolved_path, true_uid, threshold=threshold)
        
        is_fta = (score == 0.0 and not allowed)
        scores.append(-1.0 if is_fta else score)
        labels.append(1)
        pos_decisions.append(allowed)
        
        item = {'path': img_or_path, 'true_id': true_uid, 'score': score}
        if not allowed and not is_fta:
            item['status'] = 'FAIL_FR'
            false_rejects.append(item)
            if true_uid in p_ids: priority_results.append(item)
        elif allowed:
            item['status'] = 'SUCCESS'
            if true_uid in p_ids: 
                priority_results.append(item)
            elif random_demo_count < RANDOM_DEMO_LIMIT:
                priority_results.append(item)
                random_demo_count += 1
        
        if is_fta:
            fta_count += 1

    for img_or_path, claim_uid in tqdm(test_neg[:neg_limit], desc=f"Impostor {desc}"):
        total_attempts += 1
        resolved_path = _resolve(img_or_path, data_root)
        allowed, score = system.authenticate(resolved_path, claim_uid, threshold=threshold)
        
        is_fta = (score == 0.0 and not allowed)
        scores.append(-1.0 if is_fta else score)
        labels.append(0)
        neg_decisions.append(allowed)

        if allowed and not is_fta:
            # False Accept (Impostor accepted)
            item = {'path': img_or_path, 'claim_id': claim_uid, 'score': score, 'status': 'FAIL_FA'}
            false_accepts.append(item)
            if claim_uid in p_ids: priority_results.append(item)
        elif not allowed and claim_uid in p_ids:
            priority_results.append({'path': img_or_path, 'claim_id': claim_uid, 'score': score, 'status': 'REJECTED_IMPOSTOR'})
            
        if is_fta:
            fta_count += 1

    return (np.array(scores), np.array(labels), pos_decisions, neg_decisions, 
            fta_count, total_attempts, false_rejects, false_accepts, priority_results)


def run_evaluation(system, test_pos, test_neg, data_root: str, threshold: float = 0.74,
                   pos_limit: int = 500, neg_limit: int = 500, title_suffix: str = "", priority_ids=None):
    print(f"\n{'='*55}")
    print(f"  {title_suffix}")
    print(f"{'='*55}")

    scores, labels, pos_dec, neg_dec, fta_count, total_attempts, f_rejects, f_accepts, p_results = collect_scores(
        system, test_pos, test_neg, data_root, threshold, pos_limit, neg_limit, desc=title_suffix, priority_ids=priority_ids
    )

    m = compute_threshold_metrics(pos_dec, neg_dec, fta_count, total_attempts)
    print(f"\nProg decyzyjny: {threshold}")
    print(f"  FAR  = {m['far']*100:.2f}%   ({m['false_accepts']} / {len(neg_dec)})")
    print(f"  FRR  = {m['frr']*100:.2f}%   ({m['false_rejects']} / {len(pos_dec)})")
    print(f"  TAR  = {m['tar']*100:.2f}%")
    print(f"  FTA  = {m['fta']*100:.2f}%   ({fta_count} / {total_attempts})")

    fpr, tpr, thresholds, roc_auc, eer, eer_thresh = compute_roc_eer(scores, labels)

    print(f"\n  EER  = {eer*100:.4f}%   (optymalny próg ≈ {eer_thresh:.4f})")
    tar_01, thr_01 = compute_tar_at_far(fpr, tpr, thresholds, 0.1)
    tar_05, thr_05 = compute_tar_at_far(fpr, tpr, thresholds, 0.05)
    tar_01_str, thr_01_str = compute_tar_at_far(fpr, tpr, thresholds, 0.01)
    tar_001, thr_001 = compute_tar_at_far(fpr, tpr, thresholds, 0.001)

    print(f"  TAR @ FAR=0.1   = {tar_01*100:.4f}% (Thresh: {thr_01:.4f})")
    print(f"  TAR @ FAR=0.05  = {tar_05*100:.4f}% (Thresh: {thr_05:.4f})")
    print(f"  TAR @ FAR=0.01  = {tar_01_str*100:.4f}% (Thresh: {thr_01_str:.4f})")
    print(f"  TAR @ FAR=0.001 = {tar_001*100:.4f}% (Thresh: {thr_001:.4f})")

    roc_path = plot_roc(fpr, tpr, roc_auc, eer, PLOTS_DIR, title_suffix)
    frr_path = plot_far_frr_vs_threshold(fpr, tpr, thresholds, eer, eer_thresh, PLOTS_DIR, title_suffix)
    print(f"\n  Wykresy: {roc_path} | {frr_path}")

    return {"far": m["far"], "frr": m["frr"], "tar": m["tar"], "fta": m["fta"],
            "auc": roc_auc, "eer": eer, "eer_threshold": eer_thresh,
            "tar_at_far_0.1": tar_01, "thresh_at_far_0.1": thr_01,
            "tar_at_far_0.05": tar_05, "thresh_at_far_0.05": thr_05,
            "tar_at_far_0.01": tar_01_str, "thresh_at_far_0.01": thr_01_str,
            "tar_at_far_0.001": tar_001, "thresh_at_far_0.001": thr_001,
            "false_rejects": f_rejects, "false_accepts": f_accepts,
            "priority_results": p_results}


def save_metrics_to_csv(model_name, task_name, res, filename="experiments/results.csv"):
    import csv
    from datetime import datetime
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.isfile(filename)
    
    row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": model_name,
        "Task": task_name,
        "FAR (%)": f"{res['far']*100:.2f}",
        "FRR (%)": f"{res['frr']*100:.2f}",
        "TAR (%)": f"{res['tar']*100:.2f}",
        "EER (%)": f"{res['eer']*100:.4f}",
        "AUC": f"{res['auc']:.4f}",
        "TAR@FAR_0.1": f"{res['tar_at_far_0.1']*100:.4f}",
        "Thr@FAR_0.1": f"{res['thresh_at_far_0.1']:.4f}",
        "TAR@FAR_0.05": f"{res['tar_at_far_0.05']*100:.4f}",
        "Thr@FAR_0.05": f"{res['thresh_at_far_0.05']:.4f}",
        "TAR@FAR_0.01": f"{res['tar_at_far_0.01']*100:.4f}",
        "Thr@FAR_0.01": f"{res['thresh_at_far_0.01']:.4f}",
        "TAR@FAR_0.001": f"{res['tar_at_far_0.001']*100:.4f}",
        "Thr@FAR_0.001": f"{res['thresh_at_far_0.001']:.4f}"
    }

    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_visual_results(results, system, data_root, output_dir, prefix="vis"):
    """
    Saves side-by-side comparison images. Generates both 'raw' and 'aligned' versions.
    """
    os.makedirs(output_dir, exist_ok=True)
    from PIL import Image, ImageDraw, ImageOps
    
    for i, res in enumerate(results[:10]): 
        probe_path = _resolve(res['path'], data_root)
        target_id = res.get('true_id') or res.get('claim_id')
        status = res.get('status', 'Result')
        
        ref_path_rel = None
        if target_id in system.database:
            ref_path_rel = system.database[target_id]['reference_path']
        ref_path = _resolve(ref_path_rel, data_root) if ref_path_rel else None

        # Generujemy obie wersje: Raw i Aligned
        for mode in ["raw", "aligned"]:
            try:
                if mode == "aligned":
                    p_img = system.preprocessor.align_face(probe_path)
                    r_img = system.preprocessor.align_face(ref_path) if ref_path else None
                    if p_img is None: p_img = _load_img(probe_path).resize((112, 112))
                else:
                    p_img = _load_img(probe_path)
                    r_img = _load_img(ref_path) if ref_path else None
                
                w, h = p_img.size
                canvas = Image.new('RGB', (w * 2 + 10, h + 40), (255, 255, 255))
                canvas.paste(p_img, (0, 30))
                
                if r_img:
                    # Dopasowujemy referencję do rozmiaru probe (crop)
                    r_img = ImageOps.fit(r_img, (w, h), centering=(0.5, 0.5))
                    canvas.paste(r_img, (w + 10, 30))
                
                draw = ImageDraw.Draw(canvas)
                color = (0, 150, 0) if "SUCCESS" in status or "REJECTED_IMPOSTOR" in status else (200, 0, 0)
                
                # Proste nagłówki zgodnie z życzeniem
                draw.text((10, 5), "PROBE", fill=(0, 0, 0))
                draw.text((w + 10, 5), "REFERENCE", fill=(0, 0, 0))
                
                score_str = f"Score: {res['score']:.4f} ({status})"
                draw.text((w // 2, h + 30), score_str, fill=color)
                
                fname = f"{prefix}_{mode}_{i}_{target_id}_{status}.png"
                canvas.save(os.path.join(output_dir, fname))
            except Exception as e:
                print(f"Błąd zapisu [{mode}] dla {target_id}: {e}")
