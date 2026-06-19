# M3D-LaMed Impression Deployment

This repository packages the final M3D-LaMed Phi-3 ResNet50 impression model for GitHub and Netlify deployment.

## Project Contents

- `public/` contains the Netlify static site.
- `netlify/functions/` contains the serverless metadata endpoint.
- `model/merged-huggingface/` contains the final merged model configuration and supporting files using Hugging Face-compatible names.
- `model/clip-alignment/` contains the final CLIP alignment checkpoints.
- `model/moco-resnet50/` contains the final MoCo ResNet50 checkpoints.
- `evaluation/` contains final LLM evaluation metrics and per-sample outputs.
- `evaluation/clip-alignment/` contains CLIP retrieval, embedding, similarity, and training-log outputs.
- `evaluation/moco-resnet50/` contains MoCo ResNet50 training metrics.
- `training/` contains the final LLM training configuration plus CLIP alignment and MoCo ResNet50 training code.

## Deployment Notes

The model includes multi-GB binary files. GitHub requires Git LFS for these files, and GitHub LFS rejects individual objects larger than 2 GB. The final `model/merged-huggingface/model.safetensors` file is therefore intentionally excluded from this repository and should be hosted externally or restored locally before running full inference. Netlify Functions are not suitable for running this model directly because they do not provide GPU inference or enough package/runtime capacity for local 3D medical LLM inference.

For a no-cost deployment path, see `FREE_DEPLOYMENT.md`.

Recommended deployment pattern:

1. Commit this repository with Git LFS enabled.
2. Host the excluded `model.safetensors` file and inference runtime on a GPU service such as Hugging Face Inference Endpoints, RunPod, Modal, AWS SageMaker, or a private GPU server.
3. Deploy the `public/` site to Netlify.
4. Point the Netlify site or serverless functions to that external inference API.

## Local Verification

```bash
npm install
npm run verify
```

## Local Streamlit Deployment

This package includes a local Streamlit deployment app that uses the final merged LLM:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

For full setup instructions, see `STREAMLIT_DEPLOYMENT_README.md`.

## CT-CHAT Style Deployment

The original Streamlit app is preserved. A second CT-CHAT-style local UI is available:

```bash
streamlit run ct_chat_streamlit_app.py --server.port 8502
```

For details, see `CT_CHAT_DEPLOYMENT_README.md`.

## Git LFS Setup

```bash
git lfs install
git lfs track "*.safetensors" "*.pt" "*.pth" "*.bin" "*.model"
git add .gitattributes
```

## Source Model

Final source folder:

`C:\MTI Research\M3D-LaMed\lamed_phi3_resnet50_impression`

Packaged folder:

`model/merged-huggingface/`

Additional packaged research components:

- CLIP alignment training code: `training/clip-alignment/`
- CLIP alignment checkpoints: `model/clip-alignment/`
- CLIP alignment evaluation: `evaluation/clip-alignment/`
- MoCo ResNet50 training code: `training/moco-resnet50/`
- MoCo ResNet50 checkpoints: `model/moco-resnet50/`
- MoCo ResNet50 evaluation: `evaluation/moco-resnet50/`
