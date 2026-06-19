# Free Deployment Guide

You can deploy the Netlify website for free, but you cannot run the full M3D-LaMed model for free on Netlify. Netlify is good for the static frontend and lightweight serverless metadata. The model needs a GPU inference backend or a very slow CPU demo.

## Free Option 1: Netlify Website Only

Use this when you want a public project page with model information and evaluation metrics.

1. Push this folder to GitHub.
2. Connect the GitHub repository to Netlify.
3. Use these Netlify settings:
   - Build command: `npm run build`
   - Publish directory: `public`
   - Functions directory: `netlify/functions`
4. Keep the model weights in Git LFS, or remove them from the GitHub repo and upload them separately to Hugging Face.

This option costs $0 and deploys the current project page.

## Free Option 2: Hugging Face Space CPU Demo

Use this only for a slow proof-of-concept inference demo.

Free Hugging Face Spaces CPU hardware gives limited CPU and RAM. The model may load slowly or fail depending on dependency and memory behavior. GPU Spaces are paid unless you receive a community GPU grant.

## Recommended Free Setup

- Netlify: frontend and project page.
- GitHub: source code.
- Git LFS or Hugging Face Hub: model files.
- Local computer or research lab GPU: real inference endpoint during demos.

## Why Not Full Free Netlify Inference?

The final model includes large binary files and requires ML dependencies, memory, and preferably GPU acceleration. Netlify Functions are designed for lightweight serverless tasks, not long-running GPU inference for 3D medical LLMs.

