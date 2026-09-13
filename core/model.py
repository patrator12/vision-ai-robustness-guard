"""
core/model.py - Vision-Language Model Engine
Supports:
1. Google PaliGemma-3B ('google/paligemma-3b-mix-224')
2. Moondream2 ('vikhyatk/moondream2')
"""

import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    PaliGemmaForConditionalGeneration, 
    PaliGemmaProcessor
)

class VisionAI:
    def __init__(self, model_type: str = "paligemma-3b"):
        self.model_type = model_type.lower()
        self.device = self._detect_best_device()
        self.model = None
        self.tokenizer = None
        self.processor = None
        
    def _detect_best_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self, model_type: str = None):
        """Loads the requested model into memory."""
        if model_type:
            self.model_type = model_type.lower()
            
        dtype = torch.float16 if self.device in ["cuda", "mps"] else torch.float32

        if "paligemma" in self.model_type:
            model_id = "google/paligemma-3b-mix-224"
            print(f"[*] Initializing Google PaliGemma-3B ({model_id}) on {self.device}...")
            
            self.processor = PaliGemmaProcessor.from_pretrained(model_id)
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map={"": self.device} if self.device != "mps" else None
            )
            if self.device == "mps":
                self.model.to("mps")
            self.model.eval()
            print("[+] Google PaliGemma-3B loaded successfully!")

        else:
            model_id = "vikhyatk/moondream2"
            revision = "2024-08-26"
            print(f"[*] Initializing Moondream2 on {self.device}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                revision=revision,
                torch_dtype=dtype,
                device_map={"": self.device} if self.device != "mps" else None
            )
            if self.device == "mps":
                self.model.to("mps")
            self.model.eval()
            print("[+] Moondream2 loaded successfully!")

        return self

    def caption(self, image: Image.Image) -> str:
        """Generates a detailed caption for the image."""
        if self.model is None:
            self.load()

        if "paligemma" in self.model_type:
            prompt = "caption en"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=40, do_sample=False)
                # Decode only the newly generated tokens
                generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
                return self.processor.decode(generated_tokens, skip_special_tokens=True).strip()
        else:
            with torch.no_grad():
                res = self.model.caption(image)
                return res.get("caption", "").strip()

    def query(self, image: Image.Image, question: str) -> str:
        """Answers a question about the image."""
        if self.model is None:
            self.load()

        if "paligemma" in self.model_type:
            prompt = f"answer en {question}"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=40, do_sample=False)
                generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
                return self.processor.decode(generated_tokens, skip_special_tokens=True).strip()
        else:
            with torch.no_grad():
                res = self.model.query(image, question)
                return res.get("answer", "").strip()
