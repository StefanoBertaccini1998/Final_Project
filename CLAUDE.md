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

### ✅ Done (Commit 4 — classical baseline HOG + SVM)
- `src/features.py`: HOG constants (ORIENTATIONS=9, PIXELS_PER_CELL=8×8, CELLS_PER_BLOCK=2×2) + WHY documentato
- `src/models/classical.py`: SVM RBF C=1.0 + StandardScaler, `predict_proba()` aggiunto
- `notebooks/03_classical_baseline.ipynb`: dataset split stratificato 70/30, HOG visualization, training, confusion matrix
- **Risultati metal_nut**: Accuracy 83.2% | F1 0.605 | Recall difetti 46% (15/28 mancati)
- Interpretazione: SVM+HOG cattura la forma globale ma manca i difetti locali sottili — giustifica EfficientNet

### ✅ Done (Commit 5 — evaluation module)
- `src/evaluate.py`: `evaluate_classification()`, `print_results()`, `results_row()`, `compute_iou()`, `compute_dice()`
- Metriche binarie separate da weighted — recall difetti come metrica safety-critical primaria
- `results_row()` produce formato tabella per confronto finale HOG+SVM vs EfficientNet
- `notebooks/04_evaluation_module.ipynb`: demo su risultati HOG+SVM, spiegazione scelte metriche
- Smoke test verificato: output corretto su input sintetici

### ✅ Done (Commit 6 — EfficientNet-B0 fine-tuning)
- `src/dataset.py`: aggiunto `MVTecTorchDataset` — PyTorch Dataset con torchvision transforms
- `src/models/deep.py`: già completo; `DeepClassifier` verificato (328K trainable in Phase 1, 4.3M in Phase 2)
- `notebooks/05_deep_learning.ipynb`: training loop 2 fasi, training curves, evaluation, confronto baseline
- **Risultati metal_nut**: Accuracy 85.1% | F1 0.634 | Precision 1.000 | Recall difetti 46%
- Analisi: precision perfetta (zero falsi allarmi) ma recall identica a SVM — causa: class imbalance + CrossEntropyLoss non pesata
- Proposta miglioramento documentata: class-weighted loss o threshold tuning

### ✅ Done (Commit 7 — model comparison and results)
- `notebooks/06_model_comparison.ipynb`: tabella comparativa, confusion matrices affiancate, curva Precision-Recall, threshold tuning
- **Risultati finali metal_nut**:
  - HOG+SVM: Accuracy 83.2% | F1 0.605 | Recall 46.4% | Precision 86.7%
  - EfficientNet (t=0.5): Accuracy 88.1% | F1 0.727 | Recall 57.1% | Precision 100%
  - EfficientNet (t=0.3): Accuracy 89.1% | F1 **0.784** | Recall **71.4%** | Precision 87.0%
- Curva PR dimostra EfficientNet superiore a qualsiasi soglia
- Sweet spot identificato: t=0.3 (recall +14%, precision 87%)

### ✅ Done (Commit 8 — ROI zone check post-processing)
- `src/postprocessing.py`: `define_roi()`, `find_part_bbox()`, `check_roi()`, `draw_overlay()` con color coding
- `notebooks/07_postprocessing.ipynb`: ROI visualization, part localization, out-of-position simulation
- Caso simulato funziona correttamente (parte spostata → magenta OUT OF POSITION)
- Nota design: margine 20% progettato per camere wide-field; per MVTec close-up ridurre a 5%
- Concetto dimostrato e limitazione documentata per il paper

### ✅ Done (Commit 9 — per-category evaluation)
- `notebooks/08_per_category_evaluation.ipynb`: HOG+SVM su tutte e 15 le categorie MVTec AD
- **Risultati aggregati**: mean F1=0.263, solo 6/15 categorie funzionano (bottle, toothbrush, metal_nut, pill, capsule, zipper)
- 9/15 categorie F1=0.000 — textures (carpet, leather, wood, tile, grid) e oggetti complessi (cable, screw, transistor, hazelnut)
- Analisi: HOG cattura gradienti strutturali, fallisce su difetti di colore/texture sottile — motiva EfficientNet
- Confronto metal_nut: HOG F1=0.605 vs EfficientNet t=0.3 F1=0.784 incluso nel notebook
- Segue protocollo benchmark Bergmann et al. (2021)

### ✅ Done (Commit 10 — end-to-end pipeline integration)
- `src/main.py`: CLI che orchestra tutti e 4 gli stage in sequenza
- Argomenti: `--image`, `--category`, `--model` (hog|efficientnet), `--threshold`, `--checkpoint`, `--save-checkpoint`, `--roi-margin`
- HOG+SVM: train on-the-fly o carica da checkpoint (.pkl)
- EfficientNet: carica checkpoint .pt salvato da notebook 05 (training troppo lento per CLI)
- Testato: PASS su buona parte (prob=1.5%), REJECT su difettosa (prob=100%)
- `--roi-margin 0.05` per immagini MVTec close-up (default 0.20 per camere wide-field)

### ✅ Done (Commit 11 — technical analysis and README)
- `README.md`: riscritto completo — risultati reali, architettura, setup, usage CLI, notebook guide, riferimenti
- `docs/paper_draft.md`: completato con numeri reali
  - Sezione 2.5: Stage 4 ROI zone check (design decisions, limitazione MVTec close-up)
  - Sezione 3: risultati completi (metal_nut HOG vs EfficientNet, threshold sensitivity, per-category table)
  - Sezione 4: failure analysis (HOG failure modes, EfficientNet bias, ROI limitation, proposed improvements)
  - Appendix Decision Log: 14 decisioni documentate con alternatives considered

### ✅ Done (Commit 12 — EfficientNet su categorie texture)
- `notebooks/09_efficientnet_texture_categories.ipynb`: EfficientNet-B0 su 6 categorie dove HOG score F1=0.000
- Categorie: carpet, leather, wood, tile, grid, cable
- Stessa strategia two-phase di notebook 05 (10+10 epoche, lr 1e-3 / 1e-5)
- Tabella comparativa HOG vs EfficientNet + threshold tuning a t=0.3
- Runtime ~30 min (da eseguire localmente)
- Analisi: HOG fallisce su color/texture anomaly, EfficientNet generalizza grazie a feature apprese
- **Risultati reali**: leather 0.839, wood 0.824, carpet 0.821, cable 0.667, tile 0.649, grid 0.111 (mean F1=0.652)
- Checkpoint salvati in `outputs/checkpoints/efficientnet_{category}.pt` (ora inclusi nel repo)
- `.gitignore` aggiornato: checkpoint .pt e .pkl ora committati per demo standalone

### ✅ Done (Commit 13 — Grad-CAM explainability)
- `notebooks/10_gradcam_visualization.ipynb`: Grad-CAM su EfficientNet-B0 per metal_nut
- `requirements.txt`: aggiunto `grad-cam>=1.5.0` (libreria `pytorch_grad_cam` di Jacob Gildenblat)
- Target layer: `model.backbone.features[-1][0]` (Conv2d 1280ch — il wrapper Conv2dNormActivation bloccava i gradienti)
- `model.unfreeze_backbone()` necessario prima di GradCAM (freeze_backbone=True blocca i backward hook)
- 6 immagini analizzate: 2 good + bent + scratch + color + flip
- **Risultati reali**:
  - Defect probs: bent 67.2%, scratch 50.3%, color 66.7%, flip 99.7%
  - Pointing accuracy: **4/93 = 4.3%** (bent 8%, scratch 0%, color 0%, flip 8.7%)
  - Sanity check (good parts): mean concentration ratio = 5.24
  - Deletion test: scratch -50.3pp, color -66.7pp (validati); bent +32.8pp (strutturale, non locale)
- Figure salvate in `outputs/results/gradcam/` a dpi=150
- `docs/paper_draft.md` Section 3.6 aggiornata con numeri reali

### ⬜ Next Step — Commit 14 (opzionale): Gradio demo "comparativa"
- `app.py`: interfaccia web con tab HOG+SVM vs EfficientNet side-by-side
- Upload immagine + selezione categoria → entrambi i modelli classificano in parallelo
- Output: verdict (PASS/REJECT), probabilità, overlay ROI, heatmap Grad-CAM (solo EfficientNet)
- Deploy-ready per HuggingFace Spaces
- Richiede checkpoints salvati in `outputs/checkpoints/` (disponibili dopo aver eseguito notebooks 03 + 05)

---

## Commit Plan (summary)

| # | Message | Status |
|---|---|---|
| 1 | `feat: initial project setup` | ✅ |
| 2 | `feat: dataset loader and exploratory analysis` | ✅ |
| 3 | `feat: preprocessing pipeline` | ✅ |
| 4 | `feat: classical baseline HOG + SVM` | ✅ |
| 5 | `feat: evaluation module` | ✅ |
| 6 | `feat: EfficientNet fine-tuning` | ✅ |
| 7 | `feat: model comparison and results` | ✅ |
| 8 | `feat: ROI zone check post-processing` | ✅ |
| 9 | `feat: per-category evaluation (all 15 MVTec categories)` | ✅ |
| 10 | `feat: end-to-end pipeline integration` | ✅ |
| 11 | `docs: technical analysis and README` | ✅ |
| 12 | `feat: EfficientNet on texture categories` | ✅ |
| 13 | `feat: Grad-CAM explainability` | ✅ |
| 14 | `feat: Gradio demo (optional)` | ⬜ Next (opt) |

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
