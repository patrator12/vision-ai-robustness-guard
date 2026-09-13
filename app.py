"""
app.py - VisionGuard Interactive Web Application
Supports Google PaliGemma-3B & Moondream2 with real-time optical chaos controls
and hallucination mitigation.
"""

import gradio as gr
from PIL import Image
from core.chaos import apply_all_corruptions
from core.guard import VisionGuard
from core.model import VisionAI

# 1. Initialize Engines
print("[*] Initializing VisionGuard components...")
guard = VisionGuard()
# Default to Google PaliGemma-3B
ai_engine = VisionAI(model_type="paligemma-3b")

def process_pipeline(input_image, model_choice, blur_val, dark_val, noise_val, custom_question):
    """
    1. Selects and loads model (PaliGemma-3B or Moondream2)
    2. Injects optical corruptions
    3. Queries model
    4. Evaluates via VisionGuard
    """
    if input_image is None:
        return None, "Please upload or capture an image.", "", ""

    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(input_image)

    # Switch model if user selected a different one
    chosen_type = "paligemma-3b" if "PaliGemma" in model_choice else "moondream2"
    if ai_engine.model_type != chosen_type or ai_engine.model is None:
        ai_engine.load(chosen_type)

    # Step 1: Apply Corruptions
    corrupted_image = apply_all_corruptions(
        input_image, 
        blur=int(blur_val), 
        darkness=int(dark_val), 
        noise=int(noise_val)
    )

    # Step 2: Query Model
    try:
        if custom_question and custom_question.strip():
            raw_response = ai_engine.query(corrupted_image, custom_question.strip())
        else:
            raw_response = ai_engine.caption(corrupted_image)
    except Exception as e:
        raw_response = f"Model execution note: {str(e)}"

    # Step 3: Evaluate via VisionGuard
    guard_result = guard.evaluate_reliability(corrupted_image, raw_response)

    # Format Metrics Display
    metrics = guard_result["optical_metrics"]
    metrics_display = f"""
### 📊 Optical Ground-Truth Telemetry
- **Active Model:** `{model_choice}`
- **Image Sharpness:** `{metrics['sharpness']}` (Lower = Blurrier)
- **Mean Luminance:** `{metrics['mean_brightness']}` / 255.0
- **Sensor Noise Floor:** `{metrics['noise_level']}`
- **Calculated Reliability:** **{guard_result['confidence_score']}%**
- **System Risk Tier:** **{guard_result['risk_level']}**
"""

    return (
        corrupted_image,
        raw_response,
        guard_result["calibrated_output"],
        metrics_display
    )

# --- Gradio UI Layout ---
custom_css = """
#main-title { text-align: center; margin-bottom: 5px; }
#subtitle { text-align: center; color: #666; margin-bottom: 25px; }
"""

with gr.Blocks(title="VisionGuard AI", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ VisionGuard: Real-Time Hallucination Guard for Vision AI", elem_id="main-title")
    gr.Markdown(
        "### Compare Google PaliGemma-3B vs. Edge models under optical degradations (blur, darkness, sensor static), and watch VisionGuard detect and protect against hallucinations.",
        elem_id="subtitle"
    )

    with gr.Row():
        # --- Left Column: Controls ---
        with gr.Column(scale=1):
            gr.Markdown("### 📸 1. Input Image & Model Selection")
            image_input = gr.Image(type="pil", label="Upload Image or Snapshot")
            
            model_selector = gr.Dropdown(
                choices=["Google PaliGemma-3B (Flagship)", "Moondream2 (1.6B Edge Model)"],
                value="Google PaliGemma-3B (Flagship)",
                label="Select Vision AI Model"
            )

            gr.Markdown("#### 🌪️ Environmental Degradation Sliders")
            blur_slider = gr.Slider(0, 10, value=0, step=1, label="Defocus / Motion Blur (Rain / Shaky Lens)")
            dark_slider = gr.Slider(0, 10, value=0, step=1, label="Low Light / Darkness (Night Mode)")
            noise_slider = gr.Slider(0, 10, value=0, step=1, label="Sensor Static / ISO Noise (Grain)")
            
            prompt_input = gr.Textbox(
                label="Optional Prompt / Question", 
                placeholder="Leave blank for automatic caption, or ask a question like 'Is there a car?'",
                lines=1
            )
            
            run_button = gr.Button("🚀 Run Stress Test & Evaluate", variant="primary")

        # --- Right Column: Outputs ---
        with gr.Column(scale=1):
            gr.Markdown("### 👁️ 2. Corrupted Feed & Model Predictions")
            corrupted_preview = gr.Image(label="Processed Optical Feed (Fed to AI)")
            
            with gr.Row():
                with gr.Column():
                    raw_output = gr.Textbox(
                        label="❌ Standard Vision AI (Unprotected)", 
                        interactive=False, 
                        lines=4
                    )
                with gr.Column():
                    guarded_output = gr.Textbox(
                        label="🛡️ VisionGuard Protected Output", 
                        interactive=False, 
                        lines=4
                    )

            telemetry_output = gr.Markdown(label="Telemetry")

    run_button.click(
        fn=process_pipeline,
        inputs=[image_input, model_selector, blur_slider, dark_slider, noise_slider, prompt_input],
        outputs=[corrupted_preview, raw_output, guarded_output, telemetry_output]
    )

if __name__ == "__main__":
    demo.launch(share=False)
