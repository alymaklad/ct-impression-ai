# Model Files

The final merged model is stored in `merged-huggingface/`.

The filenames inside that folder intentionally keep Hugging Face conventions such as `config.json`, `model.safetensors`, `tokenizer.model`, and `generation_config.json`. These names are professional deployment names and should not be changed unless the model loading code is updated at the same time.

Large binary artifacts are tracked by Git LFS through the repository `.gitattributes` file.

## Supporting Components

- `clip-alignment/` contains the final CLIP alignment checkpoints:
  - `clip_alignment_best_model.pt`
  - `clip_alignment_final_model.pt`
- `moco-resnet50/` contains the final MoCo ResNet50 checkpoints:
  - `moco_resnet50_best_checkpoint.pth`
  - `moco_resnet50_final_checkpoint.pth`
