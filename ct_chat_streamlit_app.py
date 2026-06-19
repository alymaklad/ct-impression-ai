from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import streamlit as st

from deployment_inference import generate_impression, load_deployment_components, load_preprocessed_volume_array
from raw_ct_preprocessing import load_npy_ct_volume, preprocess_nifti_ct_volume


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model" / "merged-huggingface"

SLOPE_DEFAULT = 1.0
INTERCEPT_DEFAULT = -1024.0

WINDOWS = {
    "full": (-1000.0, 1000.0),
    "lung": (-1000.0, 400.0),
    "mediastinum": (-150.0, 250.0),
}

st.set_page_config(
    page_title="CT Impression AI",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def load_model_once():
    return load_deployment_components()


@st.cache_data(show_spinner=False)
def preprocess_uploaded_volume(file_bytes: bytes, filename: str):
    suffix = ".nii.gz" if filename.lower().endswith(".nii.gz") else ".npy"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp_path = Path(temp.name)
            temp.write(file_bytes)
            temp.flush()
            os.fsync(temp.fileno())

        if suffix == ".npy":
            normalized, display_hu = load_npy_ct_volume(temp_path)
        else:
            normalized, display_hu = preprocess_nifti_ct_volume(
                temp_path, slope=SLOPE_DEFAULT, intercept=INTERCEPT_DEFAULT
            )
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    file_id = hashlib.sha256(file_bytes).hexdigest()[:12]
    return normalized, display_hu, file_id


def preprocess_local_debug_copy(file_bytes: bytes, filename: str):
    suffix = ".nii.gz" if filename.lower().endswith(".nii.gz") else ".npy"
    debug_dir = PROJECT_ROOT / ".runtime" / "uploads"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / f"latest_upload{suffix}"
    debug_path.write_bytes(file_bytes)
    if suffix == ".npy":
        return load_npy_ct_volume(debug_path)
    return preprocess_nifti_ct_volume(debug_path, slope=SLOPE_DEFAULT, intercept=INTERCEPT_DEFAULT)


def window_slice(slice_hu: np.ndarray, window: str) -> np.ndarray:
    slice_hu = np.nan_to_num(slice_hu.astype(np.float32), nan=-1000.0, posinf=1000.0, neginf=-1000.0)
    if float(np.max(slice_hu) - np.min(slice_hu)) <= 1e-6:
        return np.zeros_like(slice_hu, dtype=np.uint8)
    low, high = WINDOWS[window]
    clipped = np.clip(slice_hu, low, high)
    scaled = (clipped - low) / max(high - low, 1.0)
    return (scaled * 255).astype(np.uint8)


st.markdown(
    """
    <style>
    .stApp {
        background: #020713;
        color: #eef6ff;
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }
    .block-container {
        max-width: 1160px;
        padding-top: 2rem;
        padding-bottom: 1.5rem;
    }
    h1 {
        color: #f8fbff;
        font-size: 1.55rem !important;
        letter-spacing: 0;
        margin-bottom: .7rem;
    }
    # .viewer-card {
    #     border: 1px solid rgba(104, 166, 255, .28);
    #     background: #071426;
    #     border-radius: 8px;
    #     padding: 18px;
    #     min-height: 620px;
    #     display: flex;
    #     align-items: center;
    #     justify-content: center;
    # }
    .viewer-card img {
        max-height: 500px;
        object-fit: contain;
        background: #000;
    }
    # .report-card {
    #     border: 1px solid rgba(104, 166, 255, .28);
    #     background: #0b1b31;
    #     border-radius: 8px;
    #     padding: 16px;
    #     margin-top: 16px;
    # }
    .small-note {
        color: rgba(226, 239, 255, .68);
        font-size: .82rem;
        margin-top: -8px;
        margin-bottom: 10px;
    }
    div[data-testid="stFileUploaderDropzone"] {
        background: #252631;
        border: 1px dashed rgba(226, 239, 255, .35);
        min-height: 132px;
    }
    .stButton > button {
        width: 100%;
        border-radius: 7px;
        background: #1f6feb;
        color: white;
        border: 1px solid #3984ff;
        min-height: 42px;
        font-weight: 700;
    }
    .stButton > button:hover {
        background: #2f7dff;
        border-color: #5b9dff;
        color: white;
    }
    div[data-baseweb="input"] > div,
    textarea {
        background: #10243d !important;
        border-color: rgba(126, 179, 255, .26) !important;
        color: #f2f7ff !important;
    }
    label, .stMarkdown, .stCaption {
        color: #eef6ff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("ct_normalized", None),
    ("ct_display_hu", None),
    ("ct_filename", None),
    ("ct_file_id", None),
    ("generated_report", ""),
    ("model_ready", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Title ──────────────────────────────────────────────────────────────────────
st.title("3D Chest CT Report Generator")

# ── Model loading — visible banner while loading, disappears when done ─────────
if not st.session_state.model_ready:
    _loading_banner = st.info("⏳  Loading LLM and 3D encoder — please wait…")
    try:
        tokenizer, llm, vision_encoder, projector, im_patch_id = load_model_once()
        st.session_state.model_ready = True
        _loading_banner.empty()
    except Exception as exc:
        _loading_banner.error("Could not load the model.")
        st.code(str(exc))
        st.stop()
else:
    tokenizer, llm, vision_encoder, projector, im_patch_id = load_model_once()

# ── Main layout ────────────────────────────────────────────────────────────────
left, right = st.columns([0.30, 0.70], gap="medium")

with left:
    st.markdown("**Upload CT Volume**")
    uploaded = st.file_uploader(
        "Upload a .npy or .nii.gz file",
        type=["npy", "gz"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        file_id = hashlib.sha256(file_bytes).hexdigest()[:12]
        if st.session_state.ct_file_id != file_id:
            try:
                _up = st.empty()
                with _up:
                    with st.spinner("Preprocessing CT volume…"):
                        try:
                            normalized, display_hu, _ = preprocess_uploaded_volume(
                                file_bytes, uploaded.name
                            )
                        except Exception:
                            normalized, display_hu = preprocess_local_debug_copy(
                                file_bytes, uploaded.name
                            )
                _up.empty()
                st.session_state.ct_normalized = normalized
                st.session_state.ct_display_hu = display_hu
                st.session_state.ct_filename = uploaded.name
                st.session_state.ct_file_id = file_id
                st.session_state.generated_report = ""
                st.success(f"Loaded {uploaded.name}")
            except Exception as exc:
                st.error("Could not load CT volume.")
                st.code(str(exc))
        else:
            st.success(f"Loaded {st.session_state.ct_filename}")

    st.divider()

    st.markdown("**Viewer**")
    window = st.radio("Window", ["full", "lung", "mediastinum"], horizontal=True, label_visibility="collapsed")

    display_hu = st.session_state.ct_display_hu
    if display_hu is not None and display_hu.shape[0] > 1:
        max_slice = int(display_hu.shape[0] - 1)
        default_slice = max_slice // 2
        slice_index = st.slider("Slice Index", 0, max_slice, default_slice)
        st.markdown(f'<div class="small-note">Shape: {display_hu.shape}</div>', unsafe_allow_html=True)
    else:
        slice_index = 0
        st.markdown('<div class="small-note">Upload a CT volume to choose a slice.</div>', unsafe_allow_html=True)

    st.divider()

    generate = st.button("Generate Report", type="primary")

with right:
    st.markdown('<div class="viewer-card">', unsafe_allow_html=True)
    if st.session_state.ct_display_hu is None:
        st.info("Upload a .npy or .nii.gz CT volume to preview slices.")
    else:
        display_volume = st.session_state.ct_display_hu
        if float(np.max(display_volume) - np.min(display_volume)) <= 1e-6:
            display_volume = ((st.session_state.ct_normalized[0] + 1.0) * 1000.0) - 1000.0
        image = window_slice(display_volume[slice_index], window)
        st.image(
            image,
            caption=f"{st.session_state.ct_filename} | slice {slice_index} | {window} window",
            width=560,
            channels="GRAY",
            clamp=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if generate:
        try:
            if st.session_state.ct_normalized is None:
                raise ValueError("Upload a .npy or .nii.gz CT volume first.")
            if llm is None:
                raise RuntimeError("Model is not loaded.")

            with st.spinner("Generating radiology report…"):
                pixel_values = load_preprocessed_volume_array(st.session_state.ct_normalized)
                st.session_state.generated_report = generate_impression(
                    tokenizer,
                    llm,
                    vision_encoder,
                    projector,
                    pixel_values,
                    im_patch_id,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                )
        except Exception as exc:
            st.session_state.generated_report = f"Inference failed:\n{exc}"

    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.markdown("**Generated Report**")
    st.text_area(
        "Generated Report",
        value=st.session_state.generated_report,
        height=180,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.caption(f"Using local model: {MODEL_DIR}")
