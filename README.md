# Smart Factory Vision Monitor

> Introduction to Computer Vision — EPICODE Institute of Technology  
> Student: Stefano Bertaccini | Exam: July 2026

**🚀 Live demo: [https://finalproject-production-78f4.up.railway.app](https://finalproject-production-78f4.up.railway.app)**  
Upload an image or pick one from the built-in MVTec gallery — HOG+SVM and EfficientNet run side-by-side with Grad-CAM overlay.

---

Automated visual defect detection for industrial parts. A fixed overhead camera
captures images of manufactured components; the system classifies each part as
**good** or **defective** and rejects parts that fall outside the expected zone.

Two pipelines are implemented and compared:
- **Classical baseline**: HOG features + SVM (fast, interpretable, no GPU needed)
- **Deep learning**: EfficientNet-B0 V2 fine-tuned via two-phase transfer learning + class-weighted loss

Dataset: [MVTec Anomaly Detection](https://www.mvtec.com/company/research/datasets/mvtec-ad)
(15 industrial categories — primary evaluation on `metal_nut`)

---

## Results

### metal_nut — HOG+SVM vs EfficientNet-B0

| Model | Accuracy | F1 (defect) | Recall (defect) | Precision (defect) |
|---|---|---|---|---|
| HOG + SVM | 83.2% | 0.605 | 46.4% | 86.7% |
| EfficientNet-B0 V1 (t=0.5) | 85.1% | 0.634 | 46.4% | 100.0% |
| EfficientNet-B0 V2 (t=0.5) ★ | **88.1%** | **0.792** | **67.9%** | 95.0% |

> ★ V2 introduces class-weighted loss `[1.0, 2.60]` to address the 2.6:1 good/defective imbalance.
> False negatives reduced from 15 → 9 vs V1.

> Recall is the primary safety-critical metric: a missed defect (False Negative)
> has higher cost than a false alarm in industrial inspection.
> Threshold t=0.3 improves recall by +25 pp over the SVM baseline.

### Per-category HOG+SVM (all 15 MVTec categories)

| Result | Value |
|---|---|
| Categories evaluated | 15/15 |
| Mean F1 (defect) | 0.263 |
| Functional categories (F1 > 0) | 6/15 |
| Failed categories (F1 = 0) | 9/15 (texture/complex) |

HOG works well on structural defects (cracks, bends, holes). It fails on
texture categories (carpet, leather, wood) where defects are color/pattern
changes without strong gradient variation — motivating EfficientNet.

---

## Architecture

```
Stage 1  src/preprocessing.py     Load -> RGB -> resize 224x224 -> normalize [0,1] -> augment
Stage 2  src/features.py          HOG (classical) | EfficientNet-B0 backbone (deep)
Stage 3  src/models/              classical.py: SVM+HOG | deep.py: EfficientNet fine-tuned
Stage 4  src/postprocessing.py    ROI zone check | NMS | morphological refinement
         src/evaluate.py          Accuracy, F1, Confusion Matrix, Precision-Recall curve
```

### EfficientNet-B0 two-phase fine-tuning

```
Phase 1 (10 epochs, lr=1e-3):  Backbone FROZEN  — trains custom head only
Phase 2 (10 epochs, lr=1e-5):  Backbone UNFROZEN — fine-tunes full network
```

Phase 1 avoids catastrophic forgetting of ImageNet features.
Phase 2 adapts the backbone to industrial texture at a low learning rate.

---

## Setup

**Requirements**: Python 3.10+, CUDA 12.x (optional, CPU also works)

```bash
git clone https://github.com/StefanoBertaccini1998/Final_Project.git
cd Final_Project

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
python src/check_setup.py      # verify environment
```

**Dataset** — not included in repo (3.4 GB):

```bash
kaggle datasets download -d ipythonx/mvtec-ad -p data/ --unzip
```

---

## Usage

### Single-image inference (CLI)

```bash
# HOG+SVM — trains on category data, then classifies (takes ~30s first run)
python src/main.py --image data/mvtec_ad/metal_nut/test/bent/000.png \
    --category metal_nut --roi-margin 0.05

# EfficientNet — requires checkpoint from notebook 05
python src/main.py --image data/mvtec_ad/metal_nut/test/bent/000.png \
    --category metal_nut --model efficientnet \
    --checkpoint outputs/checkpoints/efficientnet_metal_nut.pt

# Save HOG+SVM checkpoint for reuse (skip retraining on next call)
python src/main.py --image ... --category metal_nut \
    --save-checkpoint outputs/checkpoints/svm_metal_nut.pkl
```

Example output:
```
============================================================
Smart Factory Vision Monitor
============================================================
Image    : data\mvtec_ad\metal_nut\test\bent\000.png
Category : metal_nut
Model    : HOG  |  Threshold: 0.50

[Stage 1] Preprocessing ...
  Shape: (224, 224, 3)  dtype: float32  range: [0.07, 0.87]

[Stage 2-3] Classification (HOG) ...
  Defect probability: 100.0%  -->  DEFECTIVE

[Stage 4] ROI zone check ...
  Overlap: 100.0%  |  In ROI: True

============================================================
VERDICT: REJECT  (reason: defective — prob=100.0%)
============================================================
```

### Notebooks (recommended for exploration)

| Notebook | Stage | Content |
|---|---|---|
| `01_explore_dataset.ipynb` | Dataset | Class distribution, sample visualization |
| `02_preprocessing_visualization.ipynb` | Stage 1 | Step-by-step preprocessing walkthrough |
| `03_classical_baseline.ipynb` | Stage 2-3a | HOG visualization, SVM training, confusion matrix |
| `04_evaluation_module.ipynb` | Evaluation | Metrics explanation, precision-recall tradeoff |
| `05_deep_learning.ipynb` | Stage 2-3b | EfficientNet training curves, Phase 1 vs Phase 2 |
| `06_model_comparison.ipynb` | Results | Side-by-side confusion matrices, PR curve, threshold tuning |
| `07_postprocessing.ipynb` | Stage 4 | ROI visualization, out-of-position simulation |
| `08_per_category_evaluation.ipynb` | Benchmark | All 15 MVTec categories, Bergmann et al. protocol |

---

## Project Structure

```
Final_Project/
├── src/
│   ├── main.py              <- CLI entry point (run the full pipeline)
│   ├── dataset.py           <- MVTec AD loader (binary + multi-class)
│   ├── preprocessing.py     <- Stage 1: load, resize, normalize, augment
│   ├── features.py          <- Stage 2: HOG feature extraction
│   ├── models/
│   │   ├── classical.py     <- Stage 3a: SVM + StandardScaler
│   │   └── deep.py          <- Stage 3b: EfficientNet-B0 fine-tuned
│   ├── postprocessing.py    <- Stage 4: ROI check, NMS, morphology
│   └── evaluate.py          <- Metrics: accuracy, F1, confusion matrix
├── notebooks/               <- One notebook per pipeline stage
├── data/mvtec_ad/           <- Dataset (NOT in repo — download separately)
├── outputs/
│   ├── checkpoints/         <- Saved model weights (NOT in repo)
│   └── results/             <- Plots and metric tables
├── docs/
│   ├── technical_analysis.pdf  <- Technical analysis document (exam deliverable)
│   └── paper_draft.md          <- Source markdown
├── requirements.txt
└── SETUP.md
```

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Backbone | EfficientNet-B0 | 5.3M params vs ResNet50's 25M — less overfitting on small MVTec dataset |
| Training strategy | Two-phase fine-tuning | Phase 1 stabilizes head; Phase 2 adapts backbone at lr=1e-5 |
| Normalization | [0,1] then ImageNet mean/std in DataLoader | Preprocessing reusable for both classical and DL pipelines |
| Baseline | HOG + SVM | Required by exam; meaningful lower bound for DL comparison |
| Evaluation protocol | Per-category (15 models) | Follows MVTec AD benchmark standard (Bergmann et al., 2021) |
| Primary metric | Recall (defect class) | Industrial safety: False Negatives (missed defects) are costlier than False Positives |
| Threshold | t=0.3 for EfficientNet | Identified via PR curve — best recall/precision tradeoff for defect detection |

---

## References

- Bergmann, P. et al. *The MVTec Anomaly Detection Dataset: A Comprehensive
  Real-World Dataset for Unsupervised Anomaly Detection.*
  International Journal of Computer Vision, 2021.
- Dalal, N. & Triggs, B. *Histograms of Oriented Gradients for Human Detection.*
  CVPR, 2005.
- Tan, M. & Le, Q. *EfficientNet: Rethinking Model Scaling for Convolutional
  Neural Networks.* ICML, 2019.
