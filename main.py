# app_local.py
# Run:
#   pip install gradio fastapi pillow torch
#   python app_local.py

from pathlib import Path
import tempfile
import gradio as gr
from fastapi import FastAPI
from gradio.routes import mount_gradio_app

# ======================= LOAD MODEL SẴN =======================

print("🚀 Loading TTS model...")

from model_cpm import text_to_speech
from proccess_wav import enhance_ref_audio, transcribe_ref_audio

print("✅ TTS model loaded!")

# ======================= STYLE =======================

custom_css = """
#app-container { max-width: 1000px; margin: 0 auto; }
.gradio-container { background: radial-gradient(circle at top, #ffffff 0, #f9fafb 55%); color: #111827; }

#title-block h1 {
  font-size: 2.4rem !important;
  font-weight: 800 !important;
  background: linear-gradient(120deg, #f97316, #eab308, #22c55e);
  -webkit-background-clip: text;
  color: transparent;
  text-align: center;
}
#title-block p { text-align:center; font-size: 0.95rem; color: #6b7280; }

.sample-card {
  border-radius: 16px;
  padding: 16px;
  background: rgba(255, 255, 255, 0.96);
  border: 1px solid rgba(148, 163, 184, 0.6);
  box-shadow: 0 18px 28px rgba(148, 163, 184, 0.35);
}
"""

# ======================= INFER FUNCTIONS =======================
def clear_ref_text_on_audio_change(_):
    return ""

def infer_tts(ref_audio_path, ref_text, gen_text, steps, cfg_value):


    if not gen_text.strip():
        raise gr.Error("Please enter text content to generate voice.")

    if len(gen_text.split()) > 5000:
        raise gr.Error("Text too long (max 5000 words).")

    if not ref_audio_path:
        # 
        ref_text = None
        enhanced_ref_audio = None
    else:
        # 1) Enhance ref audio
        enhanced_ref_audio = enhance_ref_audio(ref_audio_path)

        # 2) ASR nếu thiếu ref_text
        if not ref_text or not ref_text.strip():
            ref_text = transcribe_ref_audio(enhanced_ref_audio)
            if not ref_text:
                raise gr.Error("Không nhận dạng được Reference Text.")

        if not ref_text.strip().endswith((".", ",")):
            ref_text += "."
        ref_text = ref_text.strip()
    # 3) Run TTS (MODEL ĐÃ LOAD SẴN)
    fd, out_path = tempfile.mkstemp(suffix=".wav")

    text_to_speech(
        texts=gen_text,
        prompt_wav_path=enhanced_ref_audio,
        prompt_text=ref_text,
        inference_timesteps=int(steps),
        cfg_value=float(cfg_value),
        out_path=out_path,
    )

    return out_path


def infer_ref_text_ui(ref_audio_path):
    if not ref_audio_path:
        raise gr.Error("Upload ref audio trước.")

    enhanced = enhance_ref_audio(ref_audio_path)
    text = transcribe_ref_audio(enhanced)

    if not text:
        raise gr.Error("Không nhận dạng được nội dung.")

    return text

# ======================= UI =======================

def build_demo():
    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
        with gr.Column(elem_id="app-container"):
            gr.Markdown(
                """
<div id="title-block">
<h1>🎤 ViVoxCPM – Zero-shot TTS</h1>
<p> sao chép giọng nói tức thì ⚡</p>
</div>
"""
            )

            with gr.Row():
                with gr.Column():
                    ref_audio = gr.Audio(label="🔊 Reference Voice", type="filepath")
                    ref_text = gr.Textbox(label="📝 Reference Text (optional)", lines=3)
                    ref_audio.change(
                        clear_ref_text_on_audio_change,
                        inputs=ref_audio,
                        outputs=ref_text,
                    )
                    btn_infer_text = gr.Button("✨ Infer Text từ audio (optional)")

                    gen_text = gr.Textbox(label="📝 Text to Generate", placeholder="Nhập nội dung text bạn muốn tổng hợp...", lines=6)

                    steps = gr.Slider(8, 64, value=25, step=1, label="Steps")
                    cfg_value = gr.Slider(1.5, 5, value=2.0, step=0.1, label="CFG")

                    btn_run = gr.Button("🔥 Generate Voice", variant="primary")

                with gr.Column():
                    output_audio = gr.Audio(label="🎧 Output", type="filepath")

            btn_run.click(
                infer_tts,
                inputs=[ref_audio, ref_text, gen_text, steps, cfg_value],
                outputs=output_audio,
            )

            btn_infer_text.click(
                infer_ref_text_ui,
                inputs=ref_audio,
                outputs=ref_text,
            )
        gr.HTML(
            """
            <div style="
                margin-top:20px;
                padding:16px 18px;
                border-radius:14px;
                background: #fff7ed;
                border: 1px solid #fed7aa;
                color: #374151;
            ">

            <h3 style="margin-top:0; margin-bottom:8px;">☕ Ủng hộ dự án này</h3>

            <p style="font-size:14px; line-height:1.6; margin-bottom:12px;">
            Việc huấn luyện các mô hình TTS chất lượng cao đòi hỏi tài nguyên GPU đáng kể.
            Nếu bạn thấy mô hình này hữu ích, vui lòng xem xét hỗ trợ quá trình phát triển:
            </p>

            <div style="margin-bottom:12px;">
            <a href="https://buymeacoffee.com/khanh20017n" target="_blank">
                <img
                src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Support-orange?logo=buy-me-a-coffee"
                alt="Buy Me a Coffee"
                />
            </a>
            </div>

            <img
            src="https://huggingface.co/kjanh/ViVoxCPM-1.5/resolve/main/asserts/aa8d6020dd54530a0a45.jpg"
            width="100"
            style="border-radius:8px; margin-bottom:12px;"
            />

            <p style="font-size:14px; margin-bottom:0;">
            Mọi sự ủng hộ của các bạn là niềm động lực giúp mình phát triển
            các mô hình tốt hơn trong tương lai ❤️
            </p>

            </div>
            """
        )


    return demo

# ======================= MAIN =======================

# if __name__ == "__main__":
#     import uvicorn

#     demo = build_demo()
#     app = FastAPI()
#     app = mount_gradio_app(app, demo, path="/")

#     uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    demo = build_demo()

    demo.queue(max_size=64)

    demo.launch(
        server_name="127.0.0.1",
        server_port=8386,
        share=True,
    )
