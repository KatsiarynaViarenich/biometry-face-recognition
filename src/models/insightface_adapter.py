import numpy as np
import cv2
from PIL import Image


class InsightFaceAdapter:
    def __init__(self, model_name: str = "buffalo_s", det_size: tuple = (640, 640), device: str = "cpu"):
        import insightface
        from insightface.app import FaceAnalysis

        ctx_id = 0 if device == "cuda" else -1
        self.app = FaceAnalysis(name=model_name, providers=["CUDAExecutionProvider"])
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def get_embedding(self, img_or_path) -> np.ndarray | None:
        if isinstance(img_or_path, str):
            img_bgr = cv2.imread(img_or_path)
            if img_bgr is None:
                return None
        elif isinstance(img_or_path, Image.Image):
            img_np = np.array(img_or_path.convert("RGB"))
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            return None

        faces = self.app.get(img_bgr)
        if not faces:
            return None

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = face.normed_embedding
        return emb.astype(np.float32)
