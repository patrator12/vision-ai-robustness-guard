"""
core/chaos.py - Real-World Optical Corruption Engine
Simulates real-world environmental degradation:
1. Defocus / Rain Blur
2. Night-time Low Light
3. Grainy Sensor Noise
"""

import numpy as np
import cv2
from PIL import Image, ImageEnhance

def ensure_rgb(image: Image.Image) -> Image.Image:
    """Ensures image is in 3-channel RGB format (removes alpha/grayscale issues)."""
    if image.mode != "RGB":
        return image.convert("RGB")
    return image

def apply_defocus_blur(image: Image.Image, severity: int) -> Image.Image:
    """Simulates out-of-focus camera lens, motion, or rain on windshield."""
    if severity <= 0:
        return image
    
    image = ensure_rgb(image)
    kernel_size = severity * 4 + 1
    np_img = np.array(image)
    blurred = cv2.GaussianBlur(np_img, (kernel_size, kernel_size), sigmaX=severity * 1.5)
    return Image.fromarray(blurred)

def apply_low_light(image: Image.Image, severity: int) -> Image.Image:
    """Simulates dark, night-time, or tunnel conditions."""
    if severity <= 0:
        return image
    
    image = ensure_rgb(image)
    factor = max(0.05, 1.0 - (severity * 0.095))
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def apply_sensor_noise(image: Image.Image, severity: int) -> Image.Image:
    """Simulates cheap CCTV / drone CMOS sensor static / ISO noise."""
    if severity <= 0:
        return image
    
    image = ensure_rgb(image)
    np_img = np.array(image).astype(np.float32)
    noise_sigma = severity * 12.0
    gauss = np.random.normal(0, noise_sigma, np_img.shape)
    noisy = np.clip(np_img + gauss, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def apply_all_corruptions(image: Image.Image, blur: int = 0, darkness: int = 0, noise: int = 0) -> Image.Image:
    """Applies a combination of environmental corruptions in sequence."""
    img = ensure_rgb(image.copy())
    if blur > 0:
        img = apply_defocus_blur(img, blur)
    if darkness > 0:
        img = apply_low_light(img, darkness)
    if noise > 0:
        img = apply_sensor_noise(img, noise)
    return img

def analyze_image_quality(image: Image.Image) -> dict:
    """
    Computes objective visual quality metrics:
    - Laplacian Variance (Sharpness / Blur metric)
    - Mean Luminance (Brightness metric)
    - Noise Estimate
    """
    image = ensure_rgb(image)
    np_img = np.array(image)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_brightness = float(np.mean(gray))
    
    blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_est = float(np.std(gray.astype(np.float32) - blurred_gray.astype(np.float32)))
    
    return {
        "sharpness": round(laplacian_var, 2),
        "mean_brightness": round(mean_brightness, 2),
        "noise_level": round(noise_est, 2),
        "is_severely_degraded": (laplacian_var < 50.0 or mean_brightness < 40.0 or noise_est > 35.0)
    }
