# Project Title — [DA COMPLETARE]

> Introduction to Computer Vision — EPICODE Institute of Technology  
> Final Exam Project

---

## Overview

[Descrivi brevemente il problema che risolvi e l'approccio usato]

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
```

## Run

```bash
python src/main.py --input data/sample/ --mode inference
```

## Pipeline Architecture

```
Input Image
    ↓
[1] Preprocessing (src/preprocessing.py)
    ↓
[2] Feature Extraction (src/features.py)
    ↓
[3] Model Inference (src/models/)
    ↓
[4] Post-processing (src/postprocessing.py)
    ↓
Output / Evaluation
```

## Results

| Model | Accuracy | F1-Score | Notes |
|---|---|---|---|
| Baseline (HOG + SVM) | - | - | Classical |
| Deep Learning (ResNet50) | - | - | Fine-tuned |

## Technical Analysis

See `docs/technical_analysis.pdf` for full methodology and results.

## Colab Notebook

[Link to Colab workbook — if applicable]
