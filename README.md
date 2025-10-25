## Set 1 — Foundations (3 notebooks)

### 1) Basics — PyTorch & Tensor Foundations  
**File:** `Set 1/1_Basics-Mohammadianbisheh.ipynb`  
**What I did**
- **Environment & Colab setup:** quick sanity checks and utilities.
- **Tensor fundamentals:** creation via constructors, dtypes, device moves, slicing/indexing  
  (slice / integer / boolean), reshaping, `view`/`reshape`, transpose/axis swaps.
- **Autograd demo:** build simple graphs, call `.backward()`, inspect `.grad`, and clarify
  when graphs are retained vs. freed.
- **Data pipeline:** load **MNIST** / **CIFAR-10** with `torchvision`, use `DataLoader`, apply basic
  transforms/normalization, and one-hot utilities where needed.
- **Models:** define several small networks (`SimpleModel`, `ShallowNetwork`, `DeepNetwork`,
  `WideShallowNetwork`, `BalancedNetwork`, etc.) to compare **depth vs. width**.
- **Training loop from scratch:** forward → `CrossEntropy`/softmax where appropriate → backward →
  `optimizer.step()` (SGD/Adam) → accuracy/metrics; clean separation of train/eval.
- **Visualization:** loss curves, learned function fits on toy data, and **activation statistics**
  via forward **hooks** (`capture_activation`) to study saturation and layer behavior.

**Outcomes:** a working template for PyTorch training + intuition for shapes, grads,
and how architectural choices affect optimization and activations.

---

### 2) Neural Network **from Scratch** (NumPy)  
**File:** `Set 1/2-NN_Scratch_Mohammadianbisheh.ipynb`  
**What I did**
- **Core layers (manual forward/backward):**
  - `affine_forward` / `affine_backward`
  - `relu_forward` / `relu_backward`, `sigmoid_forward` / `sigmoid_backward`
  - `mse_loss`
- **Gradient checking:** utilities (e.g., `rel_error`) to verify analytical vs. numerical grads.
- **FullyConnectedNet:** my own class supporting multi-layer MLPs with flexible depth/width,
  parameter init, full forward/backward pass, and modular loss handling.
- **Optimization:** plain SGD and **SGD+Momentum** (custom `sgd_momentum`).
- **Experiments:** train on toy tasks and a **real dataset**  
  (**California Housing** via `sklearn`) for regression—monitor train/val losses and visualize results.
- **Utilities & viz:** parameter statistics, learning curves, and a compact
  `visualize_training_results` helper.

**Outcomes:** end-to-end understanding of backprop and training **without** any DL framework;
confidence that gradients and updates are correct.

---

### 3) Optimization Playground (first-order & second-order)  
**File:** `Set 1/3-Optimization_Mohammadianbisheh.ipynb`  
**What I did**
- **Test functions (1D/2D):** Rastrigin, Schwefel, Ill-conditioned convex, **Rosenbrock**,
  **Ackley**, **Beale**, **Eggholder**.  
  Plots: 1D curves, 2D contours, and interactive 3D surfaces.
- **Optimizers implemented (update rules coded by hand):**
  - First-order: **SGD**, **Momentum**, **NAG**, **Adagrad**, **RMSprop**, **Adam**, **Nadam**, **Adadelta**
  - Second-order / quasi-Newton: **Newton**, **L-BFGS**
- **Runner & comparisons:** unified `optimize/optimize_and_plot/run_all_optimizations` interface to
  sweep optimizers and **plot loss vs. epoch**, with per-method **hyperparameter analyses**
  (step size, momentum/β’s, ε, etc.).
- **Observations:** examine curvature sensitivity, saddle behavior, and stability/step-size trade-offs
  across landscapes (e.g., Rosenbrock valleys vs. highly multimodal Rastrigin).

**Outcomes:** practical intuition for when/why different optimizers shine, plus a reusable
sandbox to test update rules on diverse landscapes.

--- 



## Set 2 — Vision Tasks (3 notebooks)

### Q1 — Image Classification
**File:** `Set 2/Q1_Classification_Mohammadianbisheh.ipynb`  
**What’s inside**
- End-to-end classification pipeline: dataset loading, train/val split, transforms & (optional) augmentations.
- Baseline network(s) with standard training loop (optimizer, LR scheduling, checkpointing).
- Metrics & diagnostics: accuracy (top-1), per-class report/confusion matrix, learning-curve plots.
- Inference & inspection utilities (e.g., sample predictions, failure cases).

**Takeaways:** how data, architecture capacity, and regularization/augmentation trade off; reading curves to debug under/overfitting.

---

### Q2 — Semantic Segmentation
**File:** `Set 2/Q2_Segmentation_Mohammadianbisheh.ipynb`  
**What’s inside**
- Semantic segmentation pipeline with paired **image–mask** loading and synchronized transforms.
- Baseline encoder–decoder model (e.g., FCN/U-Net-style) with pixel-wise loss (CE and/or Dice).
- Metrics: mean IoU (mIoU), pixel accuracy; qualitative overlays for visual inspection.
- Training/eval loops, thresholding & post-processing helpers.

**Takeaways:** setting up mask datasets correctly, balancing losses, and using IoU/Dice to evaluate structure rather than just pixel accuracy.

---

### Q3 — Object Detection
**File:** `Set 2/Q3_Object_Detection_Mohammadianbisheh.ipynb`  
**What’s inside**
- Detection dataset loader with bounding boxes (and class labels), plus box-aware augmentations.
- Baseline detector using standard PyTorch tooling (forward pass → classification + box regression losses).
- Inference steps: score filtering and Non-Max Suppression (NMS).
- Metrics & viz: mAP@IoU thresholds (where applicable) and plotted predictions with boxes/labels.

**Takeaways:** setting up detection targets, understanding the role of NMS and confidence thresholds, and interpreting mAP beyond single-image accuracy.

---

## Set 3 — Sequence Modeling & LLMs (4 notebooks)

### Q1 — Recurrent Neural Networks (RNN/LSTM/GRU)
**File:** `Set 3/Q1_RNN_Mohammadianbisheh.ipynb`  
**What’s inside**
- End-to-end sequence modeling with **RNN/LSTM/GRU**: embeddings → recurrent blocks → classifier/LM head.  
- Training details: packed sequences, teacher forcing, **truncated BPTT**, gradient clipping, dropout.  
- Tasks demonstrated: character/word-level language modeling and sequence classification.  
- Metrics & analysis: **perplexity**, accuracy, loss curves; **temperature** / **top-k** sampling for generation.

**Takeaways:** why gating (LSTM/GRU) improves long-range dependency handling; practical stability tips (init, clip, LR).

---

### Q2 — GPT-2 (Causal Language Modeling)
**File:** `Set 3/Q2_GPT2_Mohammadianbisheh.ipynb`  
**What’s inside**
- Causal LM pipeline with a **GPT-style Transformer** (tokenization/BPE, block size, causal mask).  
- Model components: multi-head self-attention, MLP w/ **GELU**, residuals, **LayerNorm**, weight tying.  
- Training: **AdamW**, cosine/linear LR schedule, gradient accumulation; evaluation via validation **perplexity**.  
- Generation: greedy vs. **top-k/top-p (nucleus)** sampling, temperature control, and repetition penalty.

**Takeaways:** how attention replaces recurrence; effects of context length, LR schedule, and sampling strategy on quality.

---

### Q3 — PEFT (LoRA / Adapters) on GPT-2
**File:** `Set 3/Q3_PEFT_Mohammadianbisheh.ipynb`  
**What’s inside**
- **Parameter-Efficient Fine-Tuning** of a pretrained GPT-2 (e.g., LoRA on attention projections).  
- Frozen base weights; train only small adapter parameters (typical: `r`, `alpha`, dropout configured).  
- Optional low-precision loading (4-/8-bit) for memory-constrained hardware.  
- Tasks: domain adaptation / sentiment-style prompts; save **adapter weights** and demonstrate **merge/unmerge**.  
- Comparisons: full FT vs. PEFT in memory/compute and validation perplexity or task score.

**Takeaways:** how PEFT achieves strong adaptation with **orders-of-magnitude fewer trainable params** and lower VRAM.

---

### Q4 — Reasoning with LLMs (Inference-time Techniques)
**File:** `Set 3/Q4_Reasoning_Mohammadianbisheh.ipynb`  
**What’s inside**
- Prompting strategies for reasoning problems: **Few-shot** prompts, **Chain-of-Thought (CoT)**, and **Self-Consistency** (sample N traces, vote).  
- Variants: Best-of-n sampling, brief **beam search**, and simple **self-refine** (reflect/rewrite).  
- Evaluation: exact-match / accuracy on small math/logic sets; ablation over temperature, sample count, and beam width.  
- Error analysis: typical failure modes (hallucinated steps, arithmetic slips) and prompt fixes.

**Takeaways:** inference-time scaling (CoT + self-consistency) can yield sizable gains without retraining; trade-offs in cost vs. accuracy.

---

## Set 4 — Generative Models (2 notebooks)

### Q1 — Variational Autoencoder (VAE)
**File:** `Set 4/VAE_Mohammadianbisheh.ipynb`  
**What’s inside**
- **Model:** encoder → \(q_\phi(z\mid x)\) (μ, logσ²), reparameterization \(z=\mu+\sigma\odot\epsilon\), decoder → \(p_\theta(x\mid z)\).
- **Objective:** ELBO with **reconstruction loss** (BCE/MSE) + **KL** term; β-VAE option for disentanglement.
- **Training:** Adam/AdamW, optional **KL warm-up** / **cyclical β**, gradient clipping.
- **Diagnostics:** reconstructions vs. inputs, **latent space traversals**, interpolation in \(z\), per-term loss curves.
- **Extras:** sampling from prior \(p(z)=\mathcal N(0,I)\), class-conditional visualizations if labels available.

**Takeaways:** how ELBO trades off fidelity vs. regularity; effect of β, latent dimensionality, and warm-up on recon quality and disentanglement.

---

### Q2 — Diffusion Models (DDPM)
**File:** `Set 4/DDPM_Mohammadianbisheh.ipynb`  
**What’s inside**
- **Forward process:** add Gaussian noise with a **β-schedule** (linear/cosine), track \(\alpha_t,\bar\alpha_t\).
- **Reverse model:** UNet-style \(\epsilon_\theta(x_t,t)\) with timestep embeddings; **simple L2 loss** to predict noise.
- **Sampling:** ancestral denoising from \(T\to0\) (DDPM), optional **DDIM**-style deterministic sampling.
- **Guidance (optional):** classifier-free guidance via conditional/unconditional batches.
- **Visualization & metrics:** noise/noise-free trajectories, sample grids over steps, loss curves; report FID/IS if computed.
- **Efficiency:** effect of number of **sampling steps (NFE)**, EMA weights, mixed precision.

**Takeaways:** why diffusion yields high-quality samples; trade-offs between schedule, NFE, and guidance strength.

---

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




