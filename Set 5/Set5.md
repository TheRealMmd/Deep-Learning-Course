## Set 5 — Self-Supervision & Text–Image Generation (3 notebooks)

### Q1 — DINO: Self-Distillation with No Labels
**File:** `Set 5/DINO_Mohammadianbisheh.ipynb`  
**What’s inside**
- **Student–Teacher** framework with EMA updates; centering & sharpening to avoid collapse.
- Backbone: ViT/ResNet; **multi-crop augmentation** pipeline for global/local views.
- Training loop (no labels), temperature scheduling, and teacher momentum.
- **Evaluation:** linear probe / k-NN on frozen features; **t-SNE/UMAP** feature visualizations.
- **Explainability:** attention rollout / class-token maps to visualize where the model looks.

**Takeaways:** how DINO learns strong visual features without labels; why augmentations, centering, and EMA matter for stability and representation quality.

---

### Q2 — Image Generation with CLIP Guidance
**File:** `Set 5/image-generation-with-clip_Mohammadianbisheh.ipynb`  
**What’s inside**
- **Text–image alignment** via CLIP: optimize an image (pixels or latent) to maximize CLIP similarity with a prompt.
- Prompt engineering: multi-prompt weighting and optional **negative prompts** (penalize undesired concepts).
- Augmentation pool during scoring (random crops/resize) to improve robustness.
- Optimization loop (Adam), learning-rate schedule, early stopping; periodic saves.
- Qualitative gallery + prompt–score curves.

**Takeaways:** CLIP can steer generation without paired supervision; prompt weights and augmentations strongly affect fidelity vs. diversity.

---

### Q3 — Stable Diffusion (Text2Img / Img2Img / Inpainting)
**File:** `Set 5/StableDiffusion_Mohammadianbisheh.ipynb`  
**What’s inside**
- **Diffusers** pipeline for Text-to-Image; controls for guidance scale, steps, sampler, and seed.
- **Image-to-Image** refinement with strength control; **inpainting** with masks.
- Optional speed/VRAM tweaks: fp16, attention slicing, xformers, EMA weights.
- Batch generation, prompt templates, and simple grid logging.
- (If included) **ControlNet** hooks for conditioning on edges/pose.

**Takeaways:** practical knobs for quality vs. speed; reproducibility with seeds; how guidance and step count shape detail and style.

---

