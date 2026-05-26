# CLAUDE.md — Smart Factory Vision Monitor
> This file is read automatically by Claude Code on startup.
> It contains all project context, current status, conventions, and next steps.
> Update the "Current Status" section after every work session.

---

## What This Project Is

**Smart Factory Vision Monitor** — a Computer Vision pipeline for automated
industrial defect detection. A fixed overhead camera captures images of
manufactured parts; the system classifies each part as good or defective
and flags any part that falls outside the expected zone (ROI).

**Course**: Introduction to Computer Vision — EPICODE Institute of Technology  
**Student**: Stefano Bertaccini  
**Exam deadline**: July 2026 (oral + practical project, 50/50)  
**Repository**: public GitHub (to be linked once created)

---

## Architecture — Four-Stage Pipeline

```
Stage 1  src/preprocessing.py    Load → RGB convert → resize 224x224 → normalize → augment
Stage 2  src/features.py         HOG (classical) | EfficientNet-B0 backbone (deep)
Stage 3  src/models/             classical.py: SVM+HOG baseline | deep.py: EfficientNet fine-tuned
Stage 4  src/postprocessing.py   ROI zone check | NMS | morphological refinement
         src/evaluate.py         Accuracy, F1, Confusion Matrix, IoU, Dice
```

Two models are intentionally implemented and compared:
- **Classical baseline**: HOG features + SVM — required by the exam to show
  understanding of traditional CV before deep learning.
- **Deep model**: EfficientNet-B0 fine-tuned — two-phase training (head only
  first, then full network).

---

## Dataset

**MVTec Anomaly Detection** — `metal_nut` category (primary target).  
Location on disk: `data/mvtec_ad/metal_nut/`  
Structure:
```
train/good/          ← normal images only (used for training)
test/good/           ← normal images (test)
test/bent/           ← defect type 1
test/color/          ← defect type 2
test/flip/           ← defect type 3
test/scratch/        ← defect type 4
```
Labels for binary classification: `good=0`, `defective=1`.  
The dataset is NOT committed to GitHub (in .gitignore). Download via:
```bash
kaggle datasets download -d ipythonx/mvtec-ad -p data/ --unzip
```

---

## Environment

- Python 3.14 + virtualenv (`venv/`)
- PyTorch 2.11.0+cu128 — GPU available: **NVIDIA GeForce RTX 5070** (12GB VRAM)
- CUDA 12.8
- All dependencies in `requirements.txt`

Activate environment:
```bash
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

Verify setup:
```bash
python src/check_setup.py
```

---

## Current Status

> Update this section at the end of every session.

### ✅ Done (Commit 1 — project setup)
- Full project structure created
- All 5 pipeline modules scaffolded with design-rationale documentation
- `dataset.py`: MVTecDataset loader (binary + multi-class)
- `preprocessing.py`: load, normalize, denoise, augment — fully documented with WHY
- `features.py`: HOG and SIFT extraction
- `models/classical.py`: SVM + StandardScaler
- `models/deep.py`: EfficientNet-B0 with two-phase fine-tuning, fully documented
- `postprocessing.py`: NMS + morphological ops
- `evaluate.py`: classification and segmentation metrics
- `check_setup.py`: environment verification script
- `notebooks/01_explore_dataset.ipynb`: dataset exploration
- `SETUP.md`, `COMMITS.md`, `.gitignore`, `docs/paper_draft.md`
- GPU confirmed working (RTX 5070 + CUDA 12.8)
- MVTec dataset downloaded and verified (15 categories)

### ✅ Done (Commit 2 — dataset loader and exploratory analysis)
- `notebooks/01_explore_dataset.ipynb` executed end-to-end
- MVTecDataset loader verified on all 15 categories (binary + multi-class)
- Good vs defective examples visualized for `metal_nut`
- Class distribution confirmed
- Example figures saved to `outputs/results/`

### ✅ Done (Commit 3 — preprocessing pipeline)
- `src/preprocessing.py`: aggiunta `preprocess()` — funzione di alto livello che orchestra load → denoise → normalize
- Pipeline testata end-to-end su immagine reale MVTec (`metal_nut/train/good/000.png`)
- Output verificato: shape `(224, 224, 3)`, dtype `float32`, range `[0.06, 0.92]`
- Smoke test integrato nel modulo (`if __name__ == "__main__"`)
- `.gitignore` aggiornato: escluso `.claude/`
- `CLAUDE.md`: aggiunte decisioni di design (per-category evaluation, Bergmann et al.)

### ⬜ Next Step — Commit 4
Implementare e validare il baseline classico:
- `src/features.py`: estrarre features HOG da tutto il dataset `metal_nut`
- `src/models/classical.py`: trainare SVM con StandardScaler
- `notebooks/02_classical_baseline.ipynb`: run completo con metriche accuracy e F1
- Risultati da loggare in `outputs/results/`

---

## Commit Plan (summary)

| # | Message | Status |
|---|---|---|
| 1 | `feat: initial project setup` | ✅ |
| 2 | `feat: dataset loader and exploratory analysis` | ✅ |
| 3 | `feat: preprocessing pipeline` | ✅ |
| 4 | `feat: classical baseline HOG + SVM` | ⬜ Next |
| 5 | `feat: evaluation module` | ⬜ |
| 6 | `feat: EfficientNet fine-tuning` | ⬜ |
| 7 | `feat: model comparison and results` | ⬜ |
| 8 | `feat: ROI zone check post-processing` | ⬜ |
| 9 | `feat: per-category evaluation (all 15 MVTec categories)` | ⬜ |
| 10 | `feat: end-to-end pipeline integration` | ⬜ |
| 11 | `docs: technical analysis and README` | ⬜ |
| 12 | `feat: error analysis by defect type (optional)` | ⬜ optional |
| 13 | `feat: Gradio demo (optional)` | ⬜ optional |

Full details in `COMMITS.md`.

---

## Coding Conventions

These apply to every file in this project. Claude Code must follow them.

### Philosophy
- **Quality over quantity**: few well-reasoned implementations beat many rushed ones.
- **Document the WHY, not the WHAT**: comments explain design decisions,
  not what Python is doing. The code already shows the what.
- **Atomic functions**: each function does exactly one thing.
- **Every decision must be defensible**: if you cannot explain why a
  parameter was chosen, it should not be in the code.

### Python style
- Type hints on all function signatures.
- Google-style docstrings: Args, Returns, Raises sections.
- Module-level docstring explaining responsibility and key design decisions.
- Constants in UPPER_SNAKE_CASE with an inline comment explaining their value.
- No magic numbers — name every constant.

### Example of acceptable comment
```python
# EfficientNet-B0 outputs 1280-dimensional feature vectors.
# This value is architecture-specific and must match the backbone choice.
in_features = 1280
```

### Example of unacceptable comment
```python
in_features = 1280  # set in_features to 1280
```

### Notebooks
Every pipeline stage has a companion notebook that:
- Visualizes the output of that stage on real images
- Prints key statistics (shape, dtype, value range, timing)
- Documents design choices with inline markdown cells
- Serves as exam documentation — a professor should be able to follow it top to bottom

Notebook naming convention:
```
01_explore_dataset.ipynb        ← Stage 0 (dataset) ✅
02_preprocessing_visualization.ipynb  ← Stage 1 (preprocessing)
03_classical_baseline.ipynb     ← Stage 2-3a (HOG + SVM)
04_deep_learning.ipynb          ← Stage 2-3b (EfficientNet)
05_model_comparison.ipynb       ← Results comparison
06_postprocessing.ipynb         ← Stage 4 (ROI zone check)
07_per_category_evaluation.ipynb ← All 15 MVTec categories
```

### Git
- Commit only working code.
- Follow the messages in COMMITS.md exactly.
- Each commit includes its companion notebook when applicable.
- Never commit: `data/`, `venv/`, `outputs/checkpoints/*.pt`, credentials.

---

## Key Design Decisions (for oral exam preparation)

| Decision | Choice | Why |
|---|---|---|
| Backbone | EfficientNet-B0 | 5.3M params vs ResNet50's 25M — less overfitting on small MVTec dataset |
| Input size | 224×224 | Standard ImageNet size — no architectural changes for transfer learning |
| Training strategy | Two-phase fine-tuning | Phase 1: head only (avoid catastrophic forgetting). Phase 2: full network at lr=1e-5 |
| Normalization | [0,1] then ImageNet mean/std in DataLoader | Keeps preprocessing reusable for both classical and DL pipelines |
| Dataset category | metal_nut | Most representative of cutting machine components |
| Augmentation | Horizontal flip only | Nuts are left-right symmetric; vertical flip disabled (parts have orientation) |
| Baseline | HOG + SVM | Required by exam; provides meaningful lower bound for DL comparison |
| Evaluation protocol | Per-category (15 models) | Follows MVTec AD benchmark standard (Bergmann et al., 2021) — avoids feature interference between heterogeneous categories |
| Secondary objective | Error analysis by defect type | If time permits — which defect types does SVM vs EfficientNet struggle with and why |

---

## Files Map

```
Final_Project/
├── CLAUDE.md                  ← YOU ARE HERE — update after each session
├── COMMITS.md                 ← commit plan with messages
├── SETUP.md                   ← environment setup guide
├── requirements.txt
├── .gitignore
├── src/
│   ├── check_setup.py         ← run first to verify environment
│   ├── dataset.py             ← MVTec loader (Stage 0)
│   ├── preprocessing.py       ← Stage 1
│   ├── features.py            ← Stage 2 (classical)
│   ├── models/
│   │   ├── classical.py       ← Stage 3a: SVM
│   │   └── deep.py            ← Stage 3b: EfficientNet
│   ├── postprocessing.py      ← Stage 4
│   ├── evaluate.py            ← metrics
│   └── main.py                ← end-to-end pipeline (to build)
├── notebooks/
│   └── 01_explore_dataset.ipynb   ← START HERE for Commit 2
├── data/
│   └── mvtec_ad/              ← NOT on GitHub
├── outputs/
│   ├── checkpoints/           ← saved model weights
│   └── results/               ← plots and metric tables
└── docs/
    └── paper_draft.md         ← technical paper, updated progressively
```
