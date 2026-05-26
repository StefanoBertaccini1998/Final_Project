# Piano Commit — Smart Factory Vision Monitor
> Segui questo piano nell'ordine: ogni commit corrisponde a un milestone funzionante.  
> Regola: committa **solo quando il codice funziona**, mai codice rotto.

---

## Come fare un commit (reminder)

```bash
git add .
git commit -m "messaggio del commit"
git push origin main
```

---

## Commit 1 — Project setup ✅ DA FARE ORA
```
feat: initial project setup

- Project structure and module layout
- requirements.txt with CPU/GPU PyTorch options
- .gitignore (excludes dataset, venv, checkpoints)
- SETUP.md with step-by-step environment guide
- check_setup.py for environment verification
```
**File da includere**: tutto tranne `data/`, `venv/`, `outputs/`

---

## Commit 2 — Dataset & exploration
```
feat: MVTec AD dataset loader and exploratory analysis

- src/dataset.py: MVTecDataset class (binary and multi-class modes)
- notebooks/01_explore_dataset.ipynb: visual exploration
- Class distribution analysis for metal_nut category
```
**Quando**: dopo aver esplorato il dataset nel notebook e visto che carica bene

---

## Commit 3 — Preprocessing pipeline
```
feat: preprocessing pipeline (Stage 1)

- src/preprocessing.py: load, resize, normalize, denoise, augment
- Added preprocess() as single entry point for both HOG and DL pipelines
- Normalization to [0,1] range; ImageNet mean/std deferred to DataLoader
- notebooks/02_preprocessing_visualization.ipynb: visual step-by-step walkthrough
```
**Quando**: dopo aver testato il preprocessing su qualche immagine

---

## Commit 4 — Classical baseline (HOG + SVM)
```
feat: classical baseline - HOG features + SVM classifier (Stage 2-3)

- src/features.py: HOG feature extraction
- src/models/classical.py: SVM with StandardScaler
- Baseline results: accuracy X%, F1 X% on metal_nut
- notebooks/03_classical_baseline.ipynb: HOG visualization + confusion matrix
```
**Quando**: dopo aver ottenuto le prime metriche del baseline

---

## Commit 5 — Evaluation framework
```
feat: evaluation module with classification metrics

- src/evaluate.py: accuracy, precision, recall, F1, confusion matrix
- IoU and Dice coefficient for future segmentation
- Pretty-print output for results
```
**Quando**: insieme o subito dopo il commit 4

---

## Commit 6 — Deep learning model
```
feat: EfficientNet-B0 fine-tuning with custom head (Stage 2-3 DL)

- src/models/deep.py: DeepClassifier with backbone + custom head
- Training loop with GPU support (RTX 5070)
- Early stopping and model checkpointing
- notebooks/04_deep_learning.ipynb: training curves + feature map visualization
```
**Quando**: dopo il primo training completato con risultati

---

## Commit 7 — Results & comparison
```
feat: model comparison and evaluation results

- Classical (HOG+SVM): accuracy X%, F1 X%
- Deep learning (EfficientNet): accuracy X%, F1 X%
- notebooks/05_model_comparison.ipynb: side-by-side confusion matrices and plots
- outputs/results/ with saved figures
```
**Quando**: dopo aver confrontato i due modelli

---

## Commit 8 — Zone check (ROI post-processing)
```
feat: ROI zone check for out-of-range detection (Stage 4)

- src/postprocessing.py: NMS and morphological refinement
- Zone of interest definition and boundary check
- notebooks/06_postprocessing.ipynb: visual overlay on detected anomalies
```
**Quando**: dopo aver implementato e testato il zone check

---

## Commit 9 — Per-category evaluation
```
feat: per-category evaluation (all 15 MVTec categories)

- Loop training and evaluation over all 15 MVTec AD categories
- Follows benchmark protocol: Bergmann et al. (2021)
- notebooks/07_per_category_evaluation.ipynb: results table + category breakdown
- outputs/results/per_category_metrics.csv
```
**Quando**: dopo aver validato il pipeline su metal_nut

---

## Commit 10 — Main pipeline integration
```
feat: end-to-end pipeline integration

- src/main.py: full pipeline from image to result
- CLI arguments for input path, mode, category
- Combined preprocessing + feature + model + postprocessing
```
**Quando**: quando l'intera pipeline gira da main.py

---

## Commit 10 — Documentation
```
docs: technical analysis document and final README

- docs/technical_analysis.pdf
- README.md: complete with results, setup, architecture diagram
- Code comments and docstrings review
```
**Quando**: verso la fine del progetto

---

## Commit 11 — (Opzionale) Gradio demo
```
feat: interactive web demo with Gradio

- app.py: Gradio interface for image upload and inference
- Real-time zone check visualization
- Deploy-ready for HuggingFace Spaces
```
**Quando**: solo se avanza tempo dopo il commit 9

---

## Stato attuale

| Commit | Status |
|---|---|
| 1 — Project setup | ⬜ Da fare |
| 2 — Dataset & exploration | ⬜ Da fare |
| 3 — Preprocessing | ⬜ Da fare |
| 4 — Classical baseline | ⬜ Da fare |
| 5 — Evaluation framework | ⬜ Da fare |
| 6 — Deep learning | ⬜ Da fare |
| 7 — Results comparison | ⬜ Da fare |
| 8 — Zone check | ⬜ Da fare |
| 9 — Pipeline integration | ⬜ Da fare |
| 10 — Documentation | ⬜ Da fare |
| 11 — Gradio demo (opz.) | ⬜ Opzionale |
