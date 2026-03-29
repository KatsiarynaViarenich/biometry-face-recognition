import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from data.dataset import CelebADataset, prepare_splits
import torchvision.transforms as transforms
from models.arcface import ArcFaceModel

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

def get_test_transforms():
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

def validate_threshold(model, val_loader, device):
    """
    Ocenia model tworząc i weryfikując pary pozytywne i negatywne w oparciu
    o dystans kosinusowy pomiędzy odrębnymi zdjęciami.
    Zwraca średni dystans par prawdziwych i fałszywych oraz sugerowany próg.
    """
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            # W trybie weryfikacji ArcFaceModel() podaje same cechy z ArcMargin
            features = model(inputs)
            embeddings.append(features.cpu())
            labels.extend(targets) # targets to Krotka stringów z DataLoader
            
    embeddings = torch.cat(embeddings)
    
    pos_sims = []
    neg_sims = []
    
    # Tworzymy proste macierze i próbkujemy
    # Ograniczenie złożoności N^2 na validation
    max_compare = 3000
    n = len(labels)
    for _ in range(max_compare):
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i == j: continue
        
        sim = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
        if labels[i] == labels[j]:
            pos_sims.append(sim)
        else:
            neg_sims.append(sim)
            
    avg_pos = np.mean(pos_sims) if pos_sims else 0.0
    avg_neg = np.mean(neg_sims) if neg_sims else 0.0
    
    # Prosty środek decyzyjny między średnimi skupieniami jako próg bazy (Threshold)
    suggested_threshold = (avg_pos + avg_neg) / 2.0
    
    return avg_pos, avg_neg, suggested_threshold

def train_model(epochs=5, batch_size=64, lr=0.01, seed=42):
    """
    Pętla pre-trainingu.
    Kod jest oparty na skrypcie z oficjalnego repozytorium ArcFace:
    https://github.com/ronghuaiyang/arcface-pytorch/blob/master/train.py
    """
    print("Przygotowanie danych...")

    splits = prepare_splits(data_dir="data/raw", num_enroll=None, seed=seed)
    train_paths, train_labels, num_classes = splits["train"]
    val_paths, val_labels = splits["val"]
    
    print(f"Liczba unikalnych klas we wnioskowaniu głównym: {num_classes}")
    
    writer = SummaryWriter('runs/arcface_experiment')
    
    train_dataset = CelebADataset("data/raw/img_align_celeba/img_align_celeba", train_paths, train_labels, transform=get_train_transforms())
    val_dataset = CelebADataset("data/raw/img_align_celeba/img_align_celeba", val_paths, val_labels, transform=get_test_transforms())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Rozpoczęcie treningu używając: {device}")
    
    model = ArcFaceModel(num_classes=num_classes, embedding_size=512, backbone_type='inception_resnet').to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    
    best_loss = float('inf')
    save_dir = "experiments/weights"
    os.makedirs(save_dir, exist_ok=True)
    
    global_step = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoka {epoch + 1}/{epochs}")
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs, _ = model(inputs, labels)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)
            global_step += 1
            
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar('Train/Epoch_Loss', epoch_loss, epoch)
        
        print(f"Epoka {epoch+1} Średni Loss: {epoch_loss:.4f}")
        
        print("Obliczanie progu powierdzenia z profilów Walidacji...")
        avg_pos, avg_neg, thresh = validate_threshold(model, val_loader, device)
        print(f"[Val] Średnie Cosine prawdziwych: {avg_pos:.4f} | intruzów: {avg_neg:.4f} => Sugerowany próg: {thresh:.4f}")
        
        writer.add_scalar('Validation/Avg_Pos_Cosine', avg_pos, epoch)
        writer.add_scalar('Validation/Avg_Neg_Cosine', avg_neg, epoch)
        writer.add_scalar('Validation/Suggested_Threshold', thresh, epoch)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_arcface.pth"))
            print("Zapisano najlepszy model!")

    writer.close()

if __name__ == "__main__":
    train_model(epochs=5, batch_size=64)
