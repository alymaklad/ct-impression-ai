import { access } from "node:fs/promises";

const requiredPaths = [
  "public/index.html",
  "public/styles.css",
  "public/app.js",
  "public/deployment-demo.mp4",
  "streamlit_app.py",
  "ct_chat_streamlit_app.py",
  "deployment_inference.py",
  "raw_ct_preprocessing.py",
  "requirements.txt",
  "STREAMLIT_DEPLOYMENT_README.md",
  "CT_CHAT_DEPLOYMENT_README.md",
  "netlify/functions/model-health.mjs",
  "model/manifest.json",
  "model/merged-huggingface/config.json",
  "model/merged-huggingface/tokenizer.model",
  "model/merged-huggingface/projector.pt",
  "model/merged-huggingface/vision_encoder_ft.pt",
  "model/clip-alignment/clip_alignment_best_model.pt",
  "model/clip-alignment/clip_alignment_final_model.pt",
  "model/moco-resnet50/moco_resnet50_best_checkpoint.pth",
  "model/moco-resnet50/moco_resnet50_final_checkpoint.pth",
  "evaluation/evaluation_results.json",
  "evaluation/clip-alignment/clip_alignment_retrieval_metrics.csv",
  "evaluation/clip-alignment/clip_alignment_training_log.csv",
  "evaluation/moco-resnet50/moco_resnet50_training_metrics.csv",
  "training/clip-alignment/clip_alignment_pilot_training.py",
  "training/clip-alignment/clip_alignment_full_training.py",
  "training/moco-resnet50/moco_resnet50_training.py",
  "training/moco-resnet50/moco3d_model.py",
  "training/training_config.json",
  "netlify.toml"
];

const missing = [];

for (const path of requiredPaths) {
  try {
    await access(path);
  } catch {
    missing.push(path);
  }
}

if (missing.length > 0) {
  console.error(`Missing required project files:\n${missing.join("\n")}`);
  process.exit(1);
}

console.log("Project verification passed.");
