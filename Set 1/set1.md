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

