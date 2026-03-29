import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from PIL import Image

class BiometricSystem:
    def __init__(self, model, preprocessor, device='cpu'):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        
        self.database = {}

    def _get_embedding(self, img_path):
        face = self.preprocessor.align_face(img_path)
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

    def enroll_user(self, user_id, image_paths):
        embeddings = []
        for path in image_paths:
            emb = self._get_embedding(path)
            if emb is not None:
                embeddings.append(emb)
                
        if not embeddings:
            return False
            
        avg_embedding = torch.mean(torch.cat(embeddings, dim=0), dim=0, keepdim=True)
        normalized_template = F.normalize(avg_embedding, p=2, dim=1)
        
        self.database[user_id] = normalized_template.cpu().numpy()
        return True
        
    def authenticate(self, image_path, user_id, threshold=0.6):
        """
        Weryfikacja tożsamości użytkownika (1:1). 
        """
        if user_id not in self.database:
            return False, 0.0 
            
        candidate_emb = self._get_embedding(image_path)
        if candidate_emb is None:
            return False, 0.0
            
        stored_emb = torch.tensor(self.database[user_id]).to(self.device)

        similarity = F.cosine_similarity(candidate_emb, stored_emb).item()
        
        return similarity >= threshold, similarity

    def identify(self, image_path, threshold=0.6):
        """
        Identyfikacja twarzy na tle bazy (1:N).
        """
        candidate_emb = self._get_embedding(image_path)
        if candidate_emb is None:
            return None, 0.0
            
        best_id = None
        best_sim = -1.0
        
        for user_id, stored_emb in self.database.items():
            stored_emb = torch.tensor(stored_emb).to(self.device)
            sim = F.cosine_similarity(candidate_emb, stored_emb).item()
            if sim > best_sim:
                best_sim = sim
                best_id = user_id
                
        if best_sim >= threshold:
            return best_id, best_sim
        return None, best_sim
