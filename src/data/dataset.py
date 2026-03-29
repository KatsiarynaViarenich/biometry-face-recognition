import os
import random
from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms

class CelebADataset(Dataset):
    """
    Obsługa zbioru CelebA oparta na oficjalnej dokumentacji PyTorch:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """
    def __init__(self, root_dir, image_paths, labels, transform=None):
        self.root_dir = Path(root_dir)
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.root_dir / self.image_paths[idx]
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
             image = Image.new('RGB', (112, 112), color='white')
             
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        return image, label

def prepare_splits(data_dir="data/raw", num_enroll=80, seed=42):
    """
    Przygotowuje 6 ściśle izolowanych zbiorów zgodnie z założeniem laboratoryjnym:
    - Train (75% tożsamości)
    - Val (10% tożsamości) -> do dobierania progu i optymalizacji
    - Test (10% tożsamości)
    - Enrolled A (Baza - max 5 klatek z wybranych 80 profili)
    - Enrolled B (Testy - reszta klatek z wybranych 80 profili)
    - Unknowns (Posłuży do testu intruzów, pochodzi bezpośrednio z Test)
    """
    random.seed(seed)
    ident_file = os.path.join(data_dir, 'identity_CelebA.txt')
    
    data = defaultdict(list)
    with open(ident_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                img, ident = parts[0], parts[1]
                data[ident].append(img)
                
    all_identities = list(data.keys())
    random.shuffle(all_identities)
    
    total = len(all_identities)
    num_train = int(0.70 * total)
    num_val = int(0.10 * total)
    num_test = int(0.10 * total)
    
    train_ids = all_identities[:num_train]
    val_ids = all_identities[num_train:num_train+num_val]
    test_ids = all_identities[num_train+num_val:num_train+num_val+num_test]
    enroll_pool = all_identities[num_train+num_val+num_test:]
    
    # Do wdrożenia używamy zadeklarowanej liczby (lub całej puli jeśli jest mniejsza)
    # Zgodnie z 10% założeniem, mamy tu ponad 1000 osób. Wymóg projektu to >= 80 profilów.
    # Wrzucamy tutaj całą 10% pule do Enrolled A i B.
    enrolled_ids = enroll_pool[:num_enroll] if num_enroll is not None else enroll_pool
    
    # ------------------ ENROLLED A / B ------------------
    enrolled_A = {}
    enrolled_B = {}
    
    for ident in enrolled_ids:
        imgs = data[ident]
        # Przyjmujemy max 5 zdjęć na profil
        enroll_count = min(5, len(imgs) - 1)
        if enroll_count <= 0:
            enroll_count = 1  # Fallback jeżeli ktoś ma np 1 zdjęcie (prawie rzadkość w CelebA)
            
        enrolled_A[ident] = imgs[:enroll_count]
        enrolled_B[ident] = imgs[enroll_count:]
        
    # ------------------ TEST (UNKNOWNS) ------------------
    test_paths, test_labels = [], []
    for ident in test_ids:
        for img in data[ident]:
            test_paths.append(img)
            test_labels.append(ident)
            
    # ------------------ VAL (DISJOINT THRESHOLD TUNE) -----
    val_paths, val_labels = [], []
    for ident in val_ids:
        for img in data[ident]:
            val_paths.append(img)
            val_labels.append(ident)
            
    # ------------------ TRAIN ------------------
    train_paths, train_labels = [], []
    # Train ID muszą być ciągłością 0..N ze względu na specyfikę funkcji wielomianowej marginu w CrossEntropy
    id_map = {ident: i for i, ident in enumerate(train_ids)}
    
    for ident in train_ids:
        for img in data[ident]:
            train_paths.append(img)
            train_labels.append(id_map[ident])
            
    return {
        "train": (train_paths, train_labels, len(train_ids)),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
        "enrolled_A": enrolled_A,
        "enrolled_B": enrolled_B
    }
