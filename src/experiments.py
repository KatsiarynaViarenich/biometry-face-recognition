import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from data.preprocess import FacePreprocessor
from data.dataset import prepare_splits, load_custom_data
from models.arcface import ArcFaceModel
from models.pretrained_inception import PretrainedInceptionAdapter
from models.insightface_adapter import InsightFaceAdapter
from system.biometrics import BiometricSystem

from evaluation.runner import run_evaluation, save_visual_results, save_metrics_to_csv
from evaluation.speed import measure_speeds
from evaluation.corruptions import add_noise_target_psnr, adjust_luminance, apply_jpeg_compression

DATA_ROOT = "data/raw/img_align_celeba/img_align_celeba"
WEIGHTS_PATH = "experiments/weights/best_arcface_1.pth"
THRESHOLD = 0.74

def resolve_path(p):
    if p.startswith("data/"):
        return p
    return os.path.join(DATA_ROOT, p)


def make_arcface_system(splits, device):
    _, _, num_classes = splits["train"]
    preprocessor = FacePreprocessor(device=device)
    model = ArcFaceModel(num_classes=num_classes, embedding_size=512, backbone_type='inception_resnet')
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=False))
        print(f"  [ArcFace] Załadowano wagi: {WEIGHTS_PATH}")
    return BiometricSystem(model=model, preprocessor=preprocessor, device=device)


def make_pretrained_system(splits, device):
    preprocessor = FacePreprocessor(device=device)
    adapter = PretrainedInceptionAdapter(device=device, pretrained='vggface2')
    system = BiometricSystem(preprocessor=preprocessor, device=device, direct_embedder=adapter)
    print("  [InceptionResNetV1] Załadowano wagi VGGFace2 (bez fine-tuningu)")
    return system


def make_insightface_system(device):
    preprocessor = FacePreprocessor(device=device)
    adapter = InsightFaceAdapter(model_name="buffalo_s", device=device)
    system = BiometricSystem(device=device, direct_embedder=adapter, preprocessor=preprocessor)
    print("  [InsightFace buffalo_s] Model załadowany")
    return system


def prepare_data():
    custom_data = load_custom_data("data/custom")
    splits = prepare_splits(data_dir="data/raw", num_enroll=80, seed=42, custom_data=custom_data)
    print(f"\n[Data] Przygotowano {len(splits['enrolled_A'])} osób do wdrożenia (w tym {len(custom_data[0])} customowe).")
    return splits


def enroll_users(system, splits):
    enrolled_ids = list(splits["enrolled_A"].keys())
    for uid in tqdm(enrolled_ids, desc="Enrollment"):
        paths = [resolve_path(p) for p in splits["enrolled_A"][uid]]
        system.enroll_user(uid, paths)
    return enrolled_ids


def build_test_pairs(splits, enrolled_ids):
    custom_ids = sorted([uid for uid in enrolled_ids if not uid.isdigit()])
    test_pos, test_neg = [], []
    
    for uid in custom_ids:
        if uid in splits["enrolled_B"]:
            for p in splits["enrolled_B"][uid]:
                test_pos.append((p, uid))
                other_uid = np.random.choice([u for u in enrolled_ids if u != uid])
                test_neg.append((p, other_uid))

    for uid, paths in splits["enrolled_B"].items():
        if uid in custom_ids: continue
        for p in paths:
            test_pos.append((p, uid))
            other_uid = np.random.choice([u for u in enrolled_ids if u != uid])
            test_neg.append((p, other_uid))

    pos_custom = [p for p in test_pos if p[1] in custom_ids]
    pos_celeba = [p for p in test_pos if p[1] not in custom_ids]
    np.random.shuffle(pos_celeba)
    test_pos = pos_custom + pos_celeba
    
    neg_custom = [p for p in test_neg if p[1] in custom_ids]
    neg_celeba = [p for p in test_neg if p[1] not in custom_ids]
    np.random.shuffle(neg_celeba)
    test_neg = neg_custom + neg_celeba
    
    return test_pos, test_neg


def load_images(pairs, limit=500):
    result = []
    for img_path, uid in pairs[:limit]:
        img = Image.open(resolve_path(img_path)).convert('RGB')
        result.append((img, uid))
    return result


def task1(system, test_pos, test_neg, prefix, priority_ids=None):
    results = run_evaluation(system, test_pos, test_neg, DATA_ROOT,
                            threshold=THRESHOLD, pos_limit=500, neg_limit=500,
                            title_suffix=f"{prefix}_Zadanie_1", priority_ids=priority_ids)
    
    fail_dir = os.path.join("experiments/failures", prefix)
    save_visual_results(results['false_rejects'], system, DATA_ROOT, os.path.join(fail_dir, "false_rejects"), prefix="FR")
    save_visual_results(results['false_accepts'], system, DATA_ROOT, os.path.join(fail_dir, "false_accepts"), prefix="FA")
    
    demo_dir = os.path.join("experiments/demo", prefix, "Zadanie_1")
    save_visual_results(results['priority_results'], system, DATA_ROOT, demo_dir, prefix="Demo")
    
    save_metrics_to_csv(prefix, "Zadanie_1", results)
    
    print(f"  Przykłady błędów zapisane w: {fail_dir}")
    print(f"  Wyniki demo grupy zapisane w: {demo_dir}")


def task2(system, test_pos, test_neg, splits, enrolled_ids, prefix, priority_ids=None):
    test_paths, _ = splits["test"]
    extra = [(p, np.random.choice(enrolled_ids)) for p in test_paths[:150]]
    neg_ext = list(test_neg[:500]) + extra
    np.random.shuffle(neg_ext)
    res = run_evaluation(system, test_pos, neg_ext, DATA_ROOT,
                   threshold=THRESHOLD, pos_limit=500, neg_limit=len(neg_ext),
                   title_suffix=f"{prefix}_Zadanie_2", priority_ids=priority_ids)
    save_metrics_to_csv(prefix, "Zadanie_2", res)


def task5(system, test_pos, splits, enrolled_ids, prefix):
    print(f"\n>>> [{prefix}] ZADANIE 5: Pomiary czasu")
    sample = resolve_path(test_pos[0][0])
    enroll_imgs = [resolve_path(p) for p in splits["enrolled_A"][enrolled_ids[0]]]
    measure_speeds(system, sample, enroll_imgs)


def task3(system, pos_imgs, neg_imgs, prefix, priority_ids=None):
    for lo, hi in [(50, 80), (40, 50), (30, 40), (20, 30), (10, 20)]:
        name = f"Noise_PSNR_{lo}-{hi}dB"
        t_pos = [(add_noise_target_psnr(img, (lo, hi)), uid) for img, uid in pos_imgs]
        t_neg = [(add_noise_target_psnr(img, (lo, hi)), uid) for img, uid in neg_imgs]
        res = run_evaluation(system, t_pos, t_neg, data_root="",
                             threshold=THRESHOLD, pos_limit=500, neg_limit=500,
                             title_suffix=f"{prefix}_{name}", priority_ids=priority_ids)
        
        fail_dir = os.path.join("experiments/failures", prefix, name)
        save_visual_results(res['false_rejects'], system, DATA_ROOT, os.path.join(fail_dir, "false_rejects"), prefix="FR")
        save_visual_results(res['false_accepts'], system, DATA_ROOT, os.path.join(fail_dir, "false_accepts"), prefix="FA")
        
        save_visual_results(res['priority_results'], system, DATA_ROOT, os.path.join("experiments/demo", prefix, name), prefix="Demo")
        save_metrics_to_csv(prefix, name, res)


def task4(system, pos_imgs, neg_imgs, prefix, priority_ids=None):
    for method, val, name in [
        ("quadratic", None,  "Quadratic"),
        ("linear",    0.5,   "Linear_1-2"),
        ("linear",    0.6,   "Linear_3-5"),
        ("linear",    0.75,  "Linear_3-4"),
        ("linear",    1.33,  "Linear_4-3"),
        ("linear",    1.5,   "Linear_3-2"),
        ("constant",  -100,  "Constant_-100"),
        ("constant",  -20,   "Constant_-20"),
        ("constant",  -10,   "Constant_-10"),
        ("constant",  30,    "Constant_+30"),
    ]:
        task_name = f"Luminance_{name}"
        t_pos = [(adjust_luminance(img, method, val), uid) for img, uid in pos_imgs]
        t_neg = [(adjust_luminance(img, method, val), uid) for img, uid in neg_imgs]
        res = run_evaluation(system, t_pos, t_neg, data_root="",
                             threshold=THRESHOLD, pos_limit=500, neg_limit=500,
                             title_suffix=f"{prefix}_{task_name}", priority_ids=priority_ids)
        
        fail_dir = os.path.join("experiments/failures", prefix, task_name)
        save_visual_results(res['false_rejects'], system, DATA_ROOT, os.path.join(fail_dir, "false_rejects"), prefix="FR")
        save_visual_results(res['false_accepts'], system, DATA_ROOT, os.path.join(fail_dir, "false_accepts"), prefix="FA")
        
        save_visual_results(res['priority_results'], system, DATA_ROOT, os.path.join("experiments/demo", prefix, task_name), prefix="Demo")
        save_metrics_to_csv(prefix, task_name, res)


def task7(system, pos_imgs, neg_imgs, prefix, priority_ids=None):
    for quality in [80, 50, 20]:
        name = f"JPEG_Q{quality}"
        t_pos = [(apply_jpeg_compression(img, quality), uid) for img, uid in pos_imgs]
        t_neg = [(apply_jpeg_compression(img, quality), uid) for img, uid in neg_imgs]
        res = run_evaluation(system, t_pos, t_neg, data_root="",
                             threshold=THRESHOLD, pos_limit=500, neg_limit=500,
                             title_suffix=f"{prefix}_{name}", priority_ids=priority_ids)
        
        fail_dir = os.path.join("experiments/failures", prefix, name)
        save_visual_results(res['false_rejects'], system, DATA_ROOT, os.path.join(fail_dir, "false_rejects"), prefix="FR")
        save_visual_results(res['false_accepts'], system, DATA_ROOT, os.path.join(fail_dir, "false_accepts"), prefix="FA")
        
        save_visual_results(res['priority_results'], system, DATA_ROOT, os.path.join("experiments/demo", prefix, name), prefix="Demo")
        save_metrics_to_csv(prefix, name, res)



def run_all_tasks(system, splits, prefix):
    print(f"\n\n{'#'*60}")
    print(f"  MODEL: {prefix}")
    print(f"{'#'*60}")

    priority_ids = sorted([uid for uid in splits["enrolled_A"].keys() if not uid.isdigit()])
    enrolled_ids = enroll_users(system, splits)
    
    db_path = f"experiments/db_{prefix}"
    system.save_db(db_path)
    system.load_db(db_path)
    
    test_pos, test_neg = build_test_pairs(splits, enrolled_ids)

    print(f"\n>>> [{prefix}] ZADANIE 1 + 6")
    task1(system, test_pos, test_neg, prefix, priority_ids=priority_ids)

    print(f"\n>>> [{prefix}] ZADANIE 2")
    task2(system, test_pos, test_neg, splits, enrolled_ids, prefix, priority_ids=priority_ids)

    task5(system, test_pos, splits, enrolled_ids, prefix)

    print(f"\n>>> [{prefix}] Ładowanie 500 obrazów do pamięci...")
    pos_imgs = load_images(test_pos, limit=500)
    neg_imgs = load_images(test_neg, limit=500)

    print(f"\n>>> [{prefix}] BASELINE (przed korupcją, punkt odniesienia)")
    res_base = run_evaluation(system, pos_imgs, neg_imgs, data_root="",
                   threshold=THRESHOLD, pos_limit=500, neg_limit=500,
                   title_suffix=f"{prefix}_Baseline_500", priority_ids=priority_ids)
    save_visual_results(res_base['priority_results'], system, DATA_ROOT, os.path.join("experiments/demo", prefix, "Baseline"), prefix="Demo")
    save_metrics_to_csv(prefix, "Baseline", res_base)

    print(f"\n>>> [{prefix}] ZADANIE 3: Szum PSNR")
    task3(system, pos_imgs, neg_imgs, prefix, priority_ids=priority_ids)

    print(f"\n>>> [{prefix}] ZADANIE 4: Luminancja")
    task4(system, pos_imgs, neg_imgs, prefix, priority_ids=priority_ids)

    print(f"\n>>> [{prefix}] ZADANIE 7: Kompresja JPEG")
    task7(system, pos_imgs, neg_imgs, prefix, priority_ids=priority_ids)



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Urządzenie: {device}")

    splits = prepare_data()

    # print("\n\n=== Inicjalizacja: ArcFace (fine-tuned) ===")
    # system_arcface = make_arcface_system(splits, device)
    # run_all_tasks(system_arcface, splits, prefix="ArcFace")

    print("\n\n=== Inicjalizacja: InceptionResNetV1 (pretrained, bez FT) ===")
    system_inception = make_pretrained_system(splits, device)
    run_all_tasks(system_inception, splits, prefix="Inception_Pretrained")

    print("\n\n=== Inicjalizacja: InsightFace buffalo_l ===")
    system_insight = make_insightface_system(device)
    run_all_tasks(system_insight, splits, prefix="InsightFace")

    print("\n\n=== WSZYSTKIE TESTY ZAKOŃCZONE ===")
