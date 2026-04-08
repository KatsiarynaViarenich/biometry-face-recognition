import torch
import torch.nn.functional as F
import numpy as np
import faiss
import os
import sqlite3
import shutil


class BiometricSystem:
    def __init__(self, model=None, preprocessor=None, device='cpu', direct_embedder=None, embedding_dim=512):
        self.device = device
        self.embedding_dim = embedding_dim
        
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.user_ids = [] 
        self.database = {} # {uid: {'embedding': np.ndarray, 'reference_path': str}}

        self._embedder = direct_embedder

        if direct_embedder is None:
            self.model = model
            self.preprocessor = preprocessor
            self.model.to(self.device)
            self.model.eval()
        else:
            self.model = None
            self.preprocessor = preprocessor
            if hasattr(direct_embedder, 'preprocessor') and direct_embedder.preprocessor is None:
                direct_embedder.preprocessor = preprocessor

    def _get_embedding(self, img_or_path):
        if self._embedder is not None:
            emb_np = self._embedder.get_embedding(img_or_path)
            if emb_np is None:
                return None
            return torch.tensor(emb_np, dtype=torch.float32).unsqueeze(0).to(self.device)

        face = self.preprocessor.align_face(img_or_path)
        if face is None:
            return None

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        face_tensor = transform(face).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(face_tensor)
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    def enroll_user(self, user_id, images_or_paths):
        embeddings = []
        for img_or_path in images_or_paths:
            emb = self._get_embedding(img_or_path)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            return False

        avg_embedding = torch.mean(torch.cat(embeddings, dim=0), dim=0, keepdim=True)
        normalized_template_np = F.normalize(avg_embedding, p=2, dim=1).cpu().numpy()
        
        self.index.add(normalized_template_np)
        self.user_ids.append(user_id)
        
        # Store metadata including a reference path for visual debugging
        ref_path = None
        for p in images_or_paths:
            if isinstance(p, str):
                ref_path = p
                break
        
        self.database[user_id] = {
            'embedding': normalized_template_np,
            'reference_path': ref_path
        }
        
        return True

    def authenticate(self, img_or_path, user_id, threshold=0.6):
        if user_id not in self.database:
            return False, 0.0

        candidate_emb = self._get_embedding(img_or_path)
        if candidate_emb is None:
            return False, 0.0

        stored_emb_np = self.database[user_id]['embedding']
        stored_emb = torch.tensor(stored_emb_np).to(self.device)
        similarity = F.cosine_similarity(candidate_emb, stored_emb).item()
        
        return similarity >= threshold, similarity

    def identify(self, img_or_path, threshold=0.6):
        candidate_emb = self._get_embedding(img_or_path)
        if candidate_emb is None:
            return None, 0.0

        if self.index.ntotal == 0:
            return None, 0.0

        query_np = candidate_emb.cpu().numpy()
        sims, indices = self.index.search(query_np, 1)
        
        best_sim = float(sims[0][0])
        best_idx = indices[0][0]

        if best_idx != -1 and best_sim >= threshold:
            return self.user_ids[best_idx], best_sim
            
        return None, best_sim

    def save_db(self, directory):
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.bin"))
        
        db_path = os.path.join(directory, "metadata.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("DROP TABLE IF EXISTS users")
        cursor.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                embedding BLOB,
                reference_path TEXT
            )
        """)
        
        for i, uid in enumerate(self.user_ids):
            data = self.database[uid]
            emb_blob = data['embedding'].tobytes()
            ref_path = data['reference_path']
            cursor.execute("INSERT INTO users (id, user_id, embedding, reference_path) VALUES (?, ?, ?, ?)", 
                           (i, uid, emb_blob, ref_path))
            
        conn.commit()
        conn.close()
        print(f"Baza (Faiss + SQL) zapisana w {directory}")

    def load_db(self, directory):
        index_path = os.path.join(directory, "index.bin")
        db_path = os.path.join(directory, "metadata.db")
        
        if not os.path.exists(index_path) or not os.path.exists(db_path):
            print(f"Błąd: Nie znaleziono bazy w {directory}")
            return False
            
        self.index = faiss.read_index(index_path)
        
        self.user_ids = []
        self.database = {}
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, user_id, embedding, reference_path FROM users ORDER BY id")
        
        for row in cursor.fetchall():
            idx, uid, emb_blob, ref_path = row
            self.user_ids.append(uid)
            emb_np = np.frombuffer(emb_blob, dtype=np.float32).reshape(1, -1)
            self.database[uid] = {
                'embedding': emb_np,
                'reference_path': ref_path
            }
            
        conn.close()
        print(f"Baza (Faiss + SQL) wczytana z {directory} (N={self.index.ntotal})")
        return True
