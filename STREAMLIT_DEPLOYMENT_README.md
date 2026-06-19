# Local Streamlit Deployment

This project includes a local Streamlit deployment app that loads the final merged LLM and generates radiology impressions from preprocessed CT volumes.

## What This Deployment Does

- Opens a local browser app.
- Loads the final merged LLM from `model/merged-huggingface/`.
- Loads the packaged 3D ResNet50 CLIP-aligned vision encoder.
- Loads the packaged image projector.
- Accepts a `.npy` CT volume with shape `(1, D, H, W)`.
- Generates a radiology impression using the final LLM.
- Shows the final evaluation metrics and required model-file status.

The app does not upload patient data anywhere. It runs from your local machine.

## Download the Project

If the project is on GitHub:

```bash
git clone <your-github-repository-url>
cd m3d-lamed-impression-deployment
```

Because the project contains large model files, install Git LFS before cloning or pulling:

```bash
git lfs install
git lfs pull
```

If you already have the local folder, open:

```text
C:\MTI Research\m3d-lamed-impression-deployment
```

## Create the Python Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Local Deployment

```bash
streamlit run streamlit_app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

Open that URL in your browser.

On Windows, if your environment reports a duplicate OpenMP runtime error, run:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
streamlit run streamlit_app.py
```

## Use the Final LLM

1. Click `Load final LLM` in the sidebar.
2. Open the `Inference` tab.
3. Enter a local `.npy` CT volume path or upload a `.npy` file.
4. Click `Generate impression`.

Expected input shape:

```text
(1, D, H, W)
```

The app clips values to `[-1, 1]`, rescales them to `[0, 1]`, resizes the volume to `(32, 128, 128)`, extracts 16 visual tokens, and sends those tokens to the final merged LLM.

## GPU Notes

Full M3D-LaMed inference is much heavier than a normal web demo because it uses large LLM weights, a 3D vision encoder, and CT-volume preprocessing.

For real inference:

- Use a machine with an NVIDIA GPU.
- Install a CUDA-compatible PyTorch build.
- Keep the Hugging Face-compatible files in `model/merged-huggingface/`.
- Run on CPU only for loading/debugging; practical inference should use CUDA.

## Main Files

- `streamlit_app.py`: local Streamlit inference app.
- `deployment_inference.py`: final LLM, vision encoder, projector, preprocessing, and generation code.
- `requirements.txt`: Python dependencies.
- `model/merged-huggingface/`: final merged LLM files.
- `model/clip-alignment/`: CLIP alignment checkpoints.
- `model/moco-resnet50/`: MoCo ResNet50 checkpoints.
- `evaluation/`: LLM, CLIP, and MoCo evaluation outputs.
