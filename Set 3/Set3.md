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

