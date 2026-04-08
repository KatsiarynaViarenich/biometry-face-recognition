import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1


_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


class PretrainedInceptionAdapter:
    def __init__(self, device: str = "cpu", pretrained: str = "vggface2"):
        self.device = device
        self.model = InceptionResnetV1(pretrained=pretrained).eval().to(device)
        self.preprocessor = None

    def get_embedding(self, img_or_path) -> np.ndarray | None:
        if self.preprocessor is not None:
            face = self.preprocessor.align_face(img_or_path)
        else:
            face = None

        if face is None:
            try:
                if isinstance(img_or_path, str):
                    face = Image.open(img_or_path).convert("RGB")
                elif isinstance(img_or_path, Image.Image):
                    face = img_or_path.convert("RGB")
                else:
                    return None
            except Exception:
                return None

        tensor = _transform(face).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model(tensor)
            emb = F.normalize(emb, p=2, dim=1)

        return emb.squeeze(0).cpu().numpy().astype(np.float32)
