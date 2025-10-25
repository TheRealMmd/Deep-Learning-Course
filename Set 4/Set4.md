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

