# Deep Learning — Advanced Course Notebooks 

## About
This repository contains my notebooks for an **advanced** Deep Learning course, organized into **five themed sets**. The sets progress from foundations and classic vision tasks to sequence modeling/LLMs, generative modeling, and self-supervised/text–image methods.  
Each set has its **own README** inside the set folder (`Set1.md`, `Set2.md`, …) with details, instructions, and notebook notes.

---

## Repository Structure 

- **Set 1 — Foundations**  
  Basics of tensors & autograd, a NumPy neural network **from scratch** (manual forward/backward + gradient checking), and an **optimization playground** comparing first/second-order methods on standard test functions.  
  _Details:_ [`Set 1/Set1.md`](Set%201/Set1.md)

- **Set 2 — Vision Tasks**  
  Three practical pipelines: **image classification**, **semantic segmentation**, and **object detection**—including data loading, training loops, metrics (e.g., accuracy, mIoU, mAP), and qualitative visualizations.  
  _Details:_ [`Set 2/Set2.md`](Set%202/Set2.md)

- **Set 3 — Sequence Modeling & LLMs**  
  **RNN/LSTM/GRU** sequence models, a **GPT-2** causal LM (training + sampling), **PEFT/LoRA** for parameter-efficient fine-tuning, and **reasoning at inference time** (CoT, self-consistency, best-of-n).  
  _Details:_ [`Set 3/Set3.md`](Set%203/Set3.md)

- **Set 4 — Generative Models**  
  **Variational Autoencoders (VAE)** with ELBO/β-VAE options and diagnostics; **Diffusion (DDPM)** with noise schedules, UNet denoiser, DDPM/DDIM sampling, and optional classifier-free guidance.  
  _Details:_ [`Set 4/Set4.md`](Set%204/Set4.md)

- **Set 5 — Self-Supervision & Text–Image Generation**  
  **DINO** self-distillation (EMA teacher, multi-crop), **CLIP-guided image optimization** from prompts, and **Stable Diffusion** (text2img/img2img/inpainting) with practical controls for quality/speed.  
  _Details:_ [`Set 5/Set5.md`](Set%205/Set5.md)

> 🔎 Each `SetX.md` explains the notebooks inside that set (what’s implemented, how to run, metrics, and key takeaways).

