from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
import streamlit as st

from deployment_inference import generate_impression, load_deployment_components, load_npy_volume


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model" / "merged-huggingface"
MANIFEST_PATH = PROJECT_ROOT / "model" / "manifest.json"
LLM_METRICS_PATH = PROJECT_ROOT / "evaluation" / "evaluation_results.json"


st.set_page_config(
    page_title="M3D-LaMed Impression Inference",
    layout="wide",
)


@st.cache_data
def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource(show_spinner=False)
def load_model_once():
    return load_deployment_components()


def format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def file_status(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "Missing", "Not found"
    if path.is_dir():
        return "Ready", "Directory"
    return "Ready", format_size(path.stat().st_size)


manifest = load_json(MANIFEST_PATH)
llm_metrics = load_json(LLM_METRICS_PATH)

st.title("M3D-LaMed Phi-3 ResNet50 Impression Inference")
st.caption("Local deployment using the final merged LLM and packaged 3D vision components.")

with st.sidebar:
    st.header("Model")
    st.write(f"Device: `{ 'CUDA' if __import__('torch').cuda.is_available() else 'CPU' }`")
    st.write(f"Model path: `{MODEL_DIR}`")
    load_now = st.button("Load final LLM", type="primary")
    st.warning(
        "Loading the final LLM can take several minutes and requires substantial RAM. "
        "CUDA is strongly recommended for inference."
    )

if load_now:
    with st.spinner("Loading final merged LLM, vision encoder, and projector..."):
        try:
            load_model_once()
            st.sidebar.success("Final LLM loaded.")
        except Exception as exc:
            st.sidebar.error("Model loading failed.")
            st.sidebar.code(str(exc))

tab_inference, tab_model, tab_usage = st.tabs(["Inference", "Model Files", "Usage"])

with tab_inference:
    st.subheader("Generate Impression from CT Volume")
    st.write("Provide a preprocessed NumPy CT volume with shape `(1, D, H, W)`.")

    input_mode = st.radio("Input method", ["Local file path", "Upload .npy file"], horizontal=True)
    local_path = ""
    uploaded_file = None

    if input_mode == "Local file path":
        local_path = st.text_input("Path to .npy CT volume", placeholder=r"C:\path\to\volume.npy")
    else:
        uploaded_file = st.file_uploader("Upload CT volume", type=["npy"])

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        max_new_tokens = st.slider("Max new tokens", 64, 512, 256, step=32)
    with col_b:
        temperature = st.slider("Temperature", 0.1, 1.2, 0.7, step=0.1)
    with col_c:
        top_p = st.slider("Top-p", 0.5, 1.0, 0.9, step=0.05)

    if st.button("Generate impression"):
        try:
            with st.spinner("Loading final model components..."):
                tokenizer, llm, vision_encoder, projector, im_patch_id = load_model_once()

            with st.spinner("Loading CT volume..."):
                if input_mode == "Local file path":
                    if not local_path:
                        raise ValueError("Enter a local .npy file path.")
                    pixel_values = load_npy_volume(Path(local_path))
                else:
                    if uploaded_file is None:
                        raise ValueError("Upload a .npy file.")
                    with tempfile.NamedTemporaryFile(suffix=".npy", delete=True) as temp_file:
                        temp_file.write(uploaded_file.read())
                        temp_file.flush()
                        pixel_values = load_npy_volume(temp_file.name)

            with st.spinner("Generating impression with the final LLM..."):
                impression = generate_impression(
                    tokenizer,
                    llm,
                    vision_encoder,
                    projector,
                    pixel_values,
                    im_patch_id,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )

            st.subheader("Generated Impression")
            st.text_area("Output", impression, height=220)
        except Exception as exc:
            st.error("Inference failed.")
            st.code(str(exc))

with tab_model:
    st.subheader("Packaged Final Model")
    st.json(
        {
            "model_name": manifest.get("model_name"),
            "base_model": manifest.get("base_model"),
            "task": manifest.get("task"),
            "architecture": manifest.get("architecture"),
            "model_directory": manifest.get("model_directory"),
        }
    )

    st.subheader("Evaluation Snapshot")
    metric_rows = [
        {"Metric": key, "Value": value}
        for key, value in llm_metrics.items()
        if isinstance(value, (int, float, str))
    ]
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True)

    st.subheader("Required File Status")
    required_paths = [
        MODEL_DIR / "config.json",
        MODEL_DIR / "model.safetensors",
        MODEL_DIR / "tokenizer.model",
        MODEL_DIR / "projector.pt",
        MODEL_DIR / "vision_encoder_ft.pt",
        PROJECT_ROOT / "model" / "clip-alignment" / "clip_alignment_best_model.pt",
    ]
    rows = []
    for path in required_paths:
        status, size = file_status(path)
        rows.append({"File": str(path.relative_to(PROJECT_ROOT)), "Status": status, "Size": size})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

with tab_usage:
    st.subheader("Run the Local Deployment")
    st.code(
        "python -m venv .venv\n"
        ".venv\\Scripts\\activate\n"
        "pip install -r requirements.txt\n"
        "streamlit run streamlit_app.py",
        language="powershell",
    )
    st.write(
        "Use a `.npy` CT volume shaped `(1, D, H, W)`. The deployment resizes it to "
        "`(1, 32, 128, 128)`, injects the 16 visual tokens, and generates an impression "
        "with the final merged LLM."
    )
