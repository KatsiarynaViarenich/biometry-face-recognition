import torch
from facenet_pytorch import MTCNN
from PIL import Image

class FacePreprocessor:
    """
    Wykorzystanie modelu MTCNN z biblioteki facenet-pytorch;
    https://github.com/timesler/facenet-pytorch/blob/master/models/mtcnn.py
    """
    def __init__(self, device='cpu', image_size=112):
        self.mtcnn = MTCNN(
            image_size=image_size, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False,
            device=device, keep_all=False
        )
        self.device = device
        
    def align_face(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            return None
            
        face_tensor = self.mtcnn(img)
        
        if face_tensor is None:
             return None
             
        face_img = face_tensor.permute(1, 2, 0).byte().cpu().numpy()
        return Image.fromarray(face_img)
