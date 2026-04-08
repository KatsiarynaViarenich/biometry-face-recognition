import time
import numpy as np

def measure_speeds(system, test_img, enroll_paths, user_id="test_perf"):
    print("\n----- Testowanie Punktu 5: Pomiary Czasu Oceny -----")
    
    _ = system.authenticate(test_img, user_id, threshold=0.5)
    
    t0 = time.perf_counter()
    system.enroll_user(user_id, enroll_paths)
    enroll_time = time.perf_counter() - t0
    print(f"  - Czas wdrażania nowego użytkownika ({len(enroll_paths)} zdjęć): {enroll_time:.4f} s")
    
    auth_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        system.authenticate(test_img, user_id, threshold=0.5)
        auth_times.append(time.perf_counter() - t0)
    
    avg_auth = np.mean(auth_times)
    print(f"  - Średni czas weryfikacji 1:1: {avg_auth:.4f} s (±{np.std(auth_times):.4f})")
    
    n_current = len(system.database)
    t0 = time.perf_counter()
    system.identify(test_img, threshold=0.5)
    ident_time = time.perf_counter() - t0
    print(f"  - Czas identyfikacji 1:N dla N={n_current}: {ident_time:.4f} s")
    
    search_overhead = ident_time - avg_auth
    if search_overhead <= 0: search_overhead = 0.000001
    
    cost_per_n = search_overhead / max(1, n_current)
    
    def estimate(N_target):
        return avg_auth + (cost_per_n * N_target)

    print(f"  - (Szacowanie) Czas dla bazy 1,000 osób:  {estimate(1000):.4f} s")
    print(f"  - (Szacowanie) Czas dla bazy 10,000 osób: {estimate(10000):.4f} s")