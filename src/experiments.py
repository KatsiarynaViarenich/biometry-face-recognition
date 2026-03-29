import sys
import os
import torch
import numpy as np
import time
from tqdm import tqdm
from data.dataset import prepare_splits
from models.arcface import ArcFaceModel
from system.biometrics import BiometricSystem

def test_far_frr(system, test_pos, test_neg, data_root, threshold=0.5):
    print("----- Testowanie Punktu 1: Prawidłowi Użytkownicy (FRR) -----")
    false_rejects = 0
    total_pos = len(test_pos)
    
    for img_name, true_uid in tqdm(test_pos[:500], desc="Weryfikacja pozytywna"):
        img_path = os.path.join(data_root, img_name)
        allowed, score = system.authenticate(img_path, true_uid, threshold=threshold)
        if not allowed:
            false_rejects += 1
            
    frr = false_rejects / min(total_pos, 500)
    print(f"False Rejection Rate (FRR) = {frr*100:.2f}% (Odpowiedź na punkt 1)")
    
    print("\n----- Testowanie Punktu 2: Intruzi zewnątrz (FPR / FAR) -----")
    false_accepts = 0
    total_neg = len(test_neg)
    
    for img_name, _ in tqdm(test_neg[:100], desc="Wbicie intruzów (FPR)"):
        img_path = os.path.join(data_root, img_name)
        matched_id, score = system.identify(img_path, threshold=threshold)
        if matched_id is not None:
            false_accepts += 1
            
    far = false_accepts / min(total_neg, 100)
    print(f"False Acceptance Rate (FAR/FPR) dla intruzów = {far*100:.2f}% (Odpowiedź na punkt 2)")
    
def measure_speeds(system, test_img, enroll_paths, user_id="test_perf"):
    print("\n----- Testowanie Punktu 5: Pomiary Czasu Oceny -----")
    
    t0 = time.time()
    system.enroll_user(user_id, enroll_paths)
    enroll_time = time.time() - t0
    print(f"Czas wdrażania nowego użytkownika (n={len(enroll_paths)} zdjęć): {enroll_time:.4f} s")
    
    t0 = time.time()
    system.authenticate(test_img, user_id, threshold=0.5)
    auth_time = time.time() - t0
    print(f"Czas weryfikacji tożsamości (1:1) z wliczonym MTCNN i normalizacją ArcFace: {auth_time:.4f} s")
    
    if len(system.database) > 80:
         print(f"Szacunkowy lub realny czas 1:N jest przy tej bazie: N/A, autoryzacja to {auth_time}s na MTCNN, matematyka to macierz O(1)")

if __name__ == "__main__":
    from data.preprocess import FacePreprocessor
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # model.load_state_dict(torch.load("experiments/weights/best_arcface.pth"))
    
    print("Skrypt przygotowany pod testy. Po wyuczeniu modelu proszę go uruchomić.")
