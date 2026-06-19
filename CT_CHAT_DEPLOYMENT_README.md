# CT Report Generator Local Deployment

This is a second Streamlit deployment that keeps the original `streamlit_app.py` and provides a CT-viewer-style report generator.

## Run

```powershell
cd "C:\MTI Research\m3d-lamed-impression-deployment"
$env:KMP_DUPLICATE_LIB_OK="TRUE"
streamlit run ct_chat_streamlit_app.py --server.port 8502
```

Open:

```text
http://127.0.0.1:8502
```

## Inputs

The app accepts:

- `.npy` files shaped `(1, D, H, W)`.
- `.nii.gz` NIfTI files using the raw CT preprocessing extracted into `raw_ct_preprocessing.py`.
- Uploads up to 600MB through `.streamlit/config.toml`.

The UI displays CT slices with three window options:

- `full`
- `lung`
- `mediastinum`

The chat interface has been removed. The app only generates a report.

## Final LLM

This UI uses the same backend as `streamlit_app.py`:

- Final merged LLM: `model/merged-huggingface/`
- CLIP-aligned 3D vision encoder: `model/clip-alignment/clip_alignment_best_model.pt`
- Projector: `model/merged-huggingface/projector.pt`
- Fine-tuned vision encoder: `model/merged-huggingface/vision_encoder_ft.pt`

## Raw CT Preprocessing

The NIfTI preprocessing code was extracted from:

```text
C:\MTI Research\dataset\preprocess_volumes.ipynb
```

and saved as:

```text
raw_ct_preprocessing.py
```
