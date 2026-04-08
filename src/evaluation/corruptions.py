import numpy as np
from PIL import Image
import io


def add_noise_target_psnr(image: Image.Image, psnr_range: tuple):
    img_np = np.array(image).astype(np.float32)
    
    target_psnr = (psnr_range[0] + psnr_range[1]) / 2.0
    
    MAX = 255.0
    mse = (MAX**2) / (10 ** (target_psnr / 10.0))
    sigma = np.sqrt(mse)
    
    noise = np.random.normal(0, sigma, img_np.shape)
    noisy_img_np = img_np + noise
    
    noisy_img_np = np.clip(np.round(noisy_img_np), 0, 255).astype(np.uint8)
    
    return Image.fromarray(noisy_img_np)

def adjust_luminance(image: Image.Image, method: str, value=None):
    img_ycbcr = image.convert('YCbCr')
    y, cb, cr = img_ycbcr.split()
    
    y_np = np.array(y).astype(np.float32)
    
    if method == "linear":
        y_np = y_np * value
    elif method == "constant":
        y_np = y_np + value
    elif method == "quadratic":
        y_np = (y_np ** 2) / 255.0
    else:
        raise ValueError(f"Nieznana metoda: {method}")
        
    y_np = np.clip(np.round(y_np), 0, 255).astype(np.uint8)
    y_new = Image.fromarray(y_np, mode='L')
    
    img_ycbcr_new = Image.merge('YCbCr', (y_new, cb, cr))
    return img_ycbcr_new.convert('RGB')

def apply_jpeg_compression(image: Image.Image, quality: int):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')