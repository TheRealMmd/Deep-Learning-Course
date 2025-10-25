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

