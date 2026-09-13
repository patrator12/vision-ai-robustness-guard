# 🛡️ VisionGuard: A Real-Time "Lie Detector" for Vision AI

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)
[![UI: Gradio](https://img.shields.io/badge/UI-Gradio%20Interactive-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> **A lightweight, real-time safety guard that stops Vision AI models from hallucinating and making up fake objects when camera feeds get dark, blurry, or noisy.**

---

## 🚨 The Problem

Vision-Language Models (like Moondream2, PaliGemma, or LLaVA) are being deployed on drones, CCTV cameras, robotics, and smartphones. 

In lab conditions with crisp, clean photos, they work great. But **the real world is messy**:
* Rainy windshields cause **defocus blur**.
* Night-time surveillance feeds suffer from **extreme low light**.
* Cheap hardware introduces **heavy camera sensor noise**.

When exposed to these bad camera conditions, **small vision AI models get confused and start confidently lying (hallucinating)**:
* They claim there's a pedestrian on an empty road.
* They identify a weapon where there is only a shadow.
* They invent objects that simply do not exist.

Existing fixes require massive cloud servers or slow down the AI by $300\%$, making them useless for real-time edge devices.

---

## 💡 The Solution: VisionGuard

**VisionGuard** is an open-source, lightweight "lie detector" for vision models. 

Instead of blindly trusting what the AI outputs, VisionGuard monitors the AI’s internal **confidence and entropy (uncertainty)** at every single word it generates. 

If the image is too blurry or dark, and the AI starts guessing, VisionGuard:
1. **Detects the uncertainty instantly** during the forward pass.
2. **Flags the hallucinated objects in RED** on the screen.
3. **Suppresses false alarms** before they trigger bad decisions.
4. Adds **less than 10% compute overhead** (runs on free/cheap hardware).

---

## ⚡ Live Comparison

| Scenario | Raw Image Input | Standard Vision AI Output | With VisionGuard 🛡️ |
| :--- | :--- | :--- | :--- |
| **Normal Day** | Clean street photo | *"A white sedan parked next to a tree."* | ✅ Confirmed accurate. |
| **Heavy Fog / Blur** | Blurred street photo | ❌ *"A red truck, two pedestrians, and a bicycle."* **(Hallucinated!)** | 🛡️ **[Warning: Optical Blur Detected]**<br>*"A vehicle [Confidence: 89%]. Other objects suppressed due to high uncertainty."* |

---

## 🎛️ Interactive Web Playground

VisionGuard comes with a built-in interactive **Gradio dashboard**:
* **Upload any image** (or use your webcam).
* **Play with Chaos Sliders:**
  * 🌫️ **Blur Slider:** Simulate rain, motion blur, or out-of-focus lenses.
  * 🌙 **Darkness Slider:** Simulate night-time and low-light surveillance.
  * 📺 **Noise Slider:** Simulate cheap, grainy camera sensors.
* **Toggle the Guard:** Watch standard AI get tricked, then flip the "Guard" switch to watch it catch its own mistakes in real-time.

---

## 🏗️ System Architecture

```
  [Camera / Image]
         │
         ▼
 🌪️ [Chaos Engine] ──► (Injects real-time Blur, Darkness, Noise)
         │
         ▼
 👁️ [Vision AI (Moondream2)] ──► Generates text description
         │
         ▼
 🛡️ [VisionGuard Filter]
         ├── Calculates token-level entropy (uncertainty)
         ├── Gated noun verification
         ▼
 🎯 [Clean, Honest Output with Uncertainty Badges]
```

---

## 🚀 Quickstart (Run It in 3 Minutes)

### Option 1: Run Free on Google Colab (Recommended)
You don't need a GPU on your laptop. Just open the notebook in Colab, choose free **T4 GPU**, and click Run:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

### Option 2: Run Locally on Your Machine

```bash
# 1. Clone the repository
git clone https://github.com/patrator12/vision-ai-robustness-guard.git
cd vision-ai-robustness-guard

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Launch the interactive web app
python app.py
```

The web interface will automatically open at `http://localhost:7860`.

---

## 📂 Project Structure

```
vision-ai-robustness-guard/
│
├── app.py                     # Interactive Gradio web application
├── core/
│   ├── model.py               # Lightweight vision AI loader (Moondream2)
│   ├── chaos.py               # Optical degradation filters (blur, noise, low light)
│   └── guard.py               # The "Lie Detector" / entropy filtering logic
├── demo_images/               # Sample real-world test images
├── requirements.txt           # Pinned Python dependencies
├── LICENSE                    # MIT License
└── README.md                  # Project documentation
```

---

## 🛠️ Tech Stack

* **AI Model:** [Moondream2](https://github.com/vikhyat/moondream) (1.6B parameter ultra-fast Vision-Language Model)
* **Framework:** PyTorch, Hugging Face Transformers
* **Image Processing:** OpenCV, Pillow, Albumentations
* **Interactive UI:** Gradio

---

## 👨‍💻 Author

Built with ❤️ by **[@patrator12](https://github.com/patrator12)**  
4th Year B.Tech (Artificial Intelligence & Machine Learning)

---

## 📄 License

This project is licensed under the **MIT License** — free for anyone to use, modify, and build upon.
