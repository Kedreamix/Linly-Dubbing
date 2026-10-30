import os
from pathlib import Path
import gradio as gr

from tools.do_everything import do_everything

SUPPORTED_LANGUAGES = [
    "বাংলা",
    "English",
    "中文",
    "हिन्दी",
    "日本語",
    "한국어",
    "Français",
    "Español",
]

LANG_TO_TTS = {
    "বাংলা": "bn",
    "English": "en",
    "中文": "zh",
    "हिन्दी": "hi",
    "日本語": "ja",
    "한국어": "ko",
    "Français": "fr",
    "Español": "es",
}


def local_video_pipeline(video_path, target_language="বাংলা", speed_up=1.0, fps=30, subtitles=True):
    if video_path is None:
        return "No video uploaded", None

    if hasattr(video_path, "name"):
        video_file = Path(video_path.name)
    else:
        video_file = Path(str(video_path))

    if not video_file.exists():
        return "Invalid video path", None

    root_folder = "videos"
    os.makedirs(root_folder, exist_ok=True)

    output_dir = os.path.join(root_folder, video_file.stem)
    os.makedirs(output_dir, exist_ok=True)

    temp_video = os.path.join(output_dir, "input.mp4")
    if not os.path.exists(temp_video):
        import shutil
        shutil.copy2(str(video_file), temp_video)

    result = do_everything(
        root_folder=root_folder,
        url=temp_video,
        translation_method="LLM",
        translation_target_language="English",
        tts_method="cosyvoice",
        tts_target_language=LANG_TO_TTS[target_language],
        voice="bn",
        subtitles=subtitles,
        speed_up=speed_up,
        fps=fps,
        background_music=None,
        bgm_volume=0.5,
        video_volume=1.0,
        target_resolution="1080p",
        max_workers=1,
        max_retries=2,
    )
    return result


with gr.Blocks(title="Linly-Dubbing Local Workflow") as demo:
    gr.Markdown("# Linly-Dubbing")
    gr.Markdown("Drag & drop local video. Only language selector is shown. AI engine is fixed to the highest-quality default.")

    with gr.Row():
        video_input = gr.Video(label="Local Video", sources=["upload"], format="mp4")
        language = gr.Dropdown(choices=SUPPORTED_LANGUAGES, value="বাংলা", label="Target Language")
        speed = gr.Slider(minimum=0.7, maximum=1.5, step=0.05, value=1.0, label="Speed")
        fps = gr.Slider(minimum=20, maximum=60, step=1, value=30, label="FPS")
        subtitles = gr.Checkbox(value=True, label="Add subtitles")

    with gr.Row():
        run_btn = gr.Button("Generate Dubbed Video")
        out_status = gr.Textbox(label="Status")
        out_video = gr.Video(label="Final Video")

    run_btn.click(
        fn=local_video_pipeline,
        inputs=[video_input, language, speed, fps, subtitles],
        outputs=[out_status, out_video],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=6006, share=True, inbrowser=True)
