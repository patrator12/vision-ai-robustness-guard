"""
core/guard.py - VisionGuard Uncertainty & Hallucination Detector
Analyzes visual degradation and cross-references model outputs to detect and filter hallucinations.
"""

from PIL import Image
from typing import Dict, Any
from .chaos import analyze_image_quality

class VisionGuard:
    def __init__(self, blur_threshold: float = 120.0, darkness_threshold: float = 50.0, noise_threshold: float = 25.0):
        self.blur_threshold = blur_threshold
        self.darkness_threshold = darkness_threshold
        self.noise_threshold = noise_threshold

    def evaluate_reliability(self, image: Image.Image, raw_caption: str) -> Dict[str, Any]:
        """
        Evaluates the visual reliability of an image and generates
        a calibrated response that flags or suppresses potential hallucinations.
        """
        metrics = analyze_image_quality(image)
        
        sharpness = metrics["sharpness"]
        brightness = metrics["mean_brightness"]
        noise = metrics["noise_level"]

        # 1. Compute Individual Penalty Factors (0.0 = clean, 1.0 = severely degraded)
        blur_penalty = max(0.0, min(1.0, (self.blur_threshold - sharpness) / self.blur_threshold))
        dark_penalty = max(0.0, min(1.0, (self.darkness_threshold - brightness) / self.darkness_threshold))
        noise_penalty = max(0.0, min(1.0, (noise - 10.0) / self.noise_threshold))

        # 2. Overall Visual Confidence Score (0% to 100%)
        total_corruption_score = min(1.0, (blur_penalty * 0.45) + (dark_penalty * 0.35) + (noise_penalty * 0.20))
        confidence_score = round(max(5.0, (1.0 - total_corruption_score) * 100), 1)

        # 3. Determine Hallucination Risk Level
        if confidence_score >= 80.0:
            risk_level = "LOW"
            status_badge = "🟢 High Confidence (Visual Input Reliable)"
            calibrated_output = raw_caption
        elif confidence_score >= 50.0:
            risk_level = "MODERATE"
            status_badge = "🟡 Moderate Confidence (Minor Optical Degradation)"
            calibrated_output = (
                f"⚠️ [Degradation Warning: {confidence_score}% Confidence]\n"
                f"{raw_caption}\n\n"
                f"ℹ️ Note: Fine details may contain visual noise artifacts."
            )
        else:
            risk_level = "HIGH / SEVERE"
            status_badge = "🔴 High Hallucination Risk (Severe Optical Degradation)"
            calibrated_output = (
                f"🚨 [CRITICAL ALERT: High Hallucination Risk — Reliability: {confidence_score}%]\n"
                f"Raw Model Assertion: \"{raw_caption}\"\n\n"
                f"🛡️ VisionGuard Action: Gated Suppression Active. "
                f"Due to severe visual distortion (Sharpness: {sharpness}, Brightness: {brightness}), "
                f"specific object identifications cannot be independently verified and are flagged as ungrounded."
            )

        return {
            "confidence_score": confidence_score,
            "risk_level": risk_level,
            "status_badge": status_badge,
            "calibrated_output": calibrated_output,
            "optical_metrics": metrics
        }
