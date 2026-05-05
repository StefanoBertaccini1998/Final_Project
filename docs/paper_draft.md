# Smart Factory Vision Monitor
## Technical Analysis Document — Draft in Progress
> EPICODE Institute of Technology — Introduction to Computer Vision  
> Author: Stefano Bertaccini  
> Status: 🔄 Updated progressively alongside development  
> Target: PDF, max 10 pages

---

> **Come usare questo documento**  
> Ogni sezione ha uno stato: ✅ Completo / 🔄 In progress / ⬜ Da scrivere.  
> Compila ogni sezione nel momento in cui il codice corrispondente è pronto.  
> Non aspettare la fine — il paper cresce con il progetto.

---

## 1. Problem Statement ✅

### 1.1 Context and Motivation

Industrial quality control is a critical bottleneck in modern manufacturing.
Traditionally, visual inspection of produced parts is performed manually by
trained operators — a process that is slow, expensive, and subject to human
fatigue. A single production line can generate hundreds of parts per hour,
making 100% manual coverage impractical.

Computer vision offers an opportunity to automate this process: a fixed camera
above a workstation can continuously capture images of produced parts and flag
anomalies in real time, without interrupting the production flow.

### 1.2 Problem Definition

This project addresses the task of **automated visual defect detection** in
industrial components. Given an image of a manufactured part captured by a
fixed overhead camera, the system must:

1. **Classify** the part as *good* or *defective* (binary classification).
2. **Identify** the spatial region of interest (ROI) where the part is expected
   to appear, and alert if the part is outside the expected zone (out-of-range
   detection).

### 1.3 Why This Problem is Relevant

- Manufacturing defects cost the global industry an estimated $300B annually
  (source: McKinsey, 2022).
- Early detection at the production stage — before assembly — drastically
  reduces recall costs.
- The solution generalizes: the same pipeline can monitor different part
  categories by retraining only the classification head, with no architectural
  changes.

### 1.4 Scope and Limitations

- **In scope**: binary classification (good/defective) on static images.
- **Out of scope**: real-time video stream processing, 3D defect localization,
  multi-part simultaneous inspection.
- **Dataset constraint**: we use the MVTec Anomaly Detection benchmark as a
  proxy for real factory data, using the `metal_nut` category as primary
  evaluation target.

---

## 2. Methodology 🔄

### 2.1 Pipeline Architecture

The system is organized as a four-stage sequential pipeline:

```
Stage 1: Data Acquisition & Preprocessing
          ↓
Stage 2: Feature Engineering / Representation
          ↓
Stage 3: Core Classification Model
          ↓
Stage 4: Post-processing & Zone Check
```

Each stage is implemented as an independent Python module, following the
Single Responsibility Principle. This modularity enables independent testing
of each component and facilitates future extension (e.g., replacing the
backbone without touching preprocessing).

---

### 2.2 Stage 1 — Preprocessing (`src/preprocessing.py`) ✅

**Input**: raw RGB image from disk (variable resolution).  
**Output**: normalized float32 array of shape (224, 224, 3).

#### Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Target resolution | 224 × 224 px | Standard input size for ImageNet-pretrained models; no architectural changes needed for transfer learning |
| Color space | RGB | Matches torchvision convention; avoids silent BGR→RGB bugs downstream |
| Normalization range | [0, 1] | Shared between classical and deep pipelines; ImageNet mean/std applied separately in DataLoader |
| Noise reduction | Gaussian blur, σ=0, k=3 | MVTec images have sensor noise; k=3 removes noise without blurring texture boundaries |
| Augmentation | Horizontal flip only | Metal nuts are left-right symmetric; vertical flip disabled as parts have orientation |

---

### 2.3 Stage 2 — Feature Engineering (`src/features.py`) 🔄

Two parallel feature extraction pipelines are implemented for comparison:

#### 2.3.1 Handcrafted Features — HOG

**Histogram of Oriented Gradients (HOG)** captures the distribution of edge
directions across local regions. For industrial surfaces, edges and texture
gradients are the primary visual cues that distinguish defective from normal
regions.

Parameters chosen:
- Orientations: 9 bins (standard, 40° resolution per bin)
- Pixels per cell: 8×8 (balances spatial resolution vs descriptor length)
- Cells per block: 2×2 (L2 normalization across blocks improves illumination invariance)

*[Add HOG visualization here once notebook 02 is complete]*

#### 2.3.2 Learned Features — EfficientNet-B0 Backbone

*[Complete after training — add feature map visualizations]*

---

### 2.4 Stage 3 — Core Model ✅

Two models are implemented and compared:

#### 2.4.1 Classical Baseline: SVM on HOG Features (`src/models/classical.py`)

- Feature vector: HOG descriptor, flattened to 1D
- Classifier: Support Vector Machine with RBF kernel, `C=1.0`
- Preprocessing: StandardScaler (zero-mean, unit-variance per feature)
- Rationale: SVM with RBF kernel is the standard baseline for texture
  classification. It provides a meaningful performance lower bound for
  the deep learning model.

#### 2.4.2 Deep Learning: EfficientNet-B0 Fine-tuned (`src/models/deep.py`)

- Backbone: EfficientNet-B0, pretrained on ImageNet (5.3M parameters)
- Custom head: Linear(1280→256) → ReLU → Dropout(0.4) → Linear(256→2)
- Training strategy: two-phase fine-tuning (head only → full network)

**Why EfficientNet-B0 over ResNet50?**

EfficientNet-B0 achieves comparable accuracy to ResNet50 with 4.7× fewer
parameters. On small datasets like MVTec (≈300 training samples per category),
fewer parameters reduce the risk of overfitting. Additionally, EfficientNet's
compound scaling produces richer texture representations, which are more
relevant to defect detection than object-level features.

**Why transfer learning instead of training from scratch?**

The MVTec training set contains only normal (defect-free) images — typically
200–400 samples. Training a deep CNN from scratch on this volume would result
in severe underfitting. Transfer learning from ImageNet provides pretrained
low-level features (edges, textures, gradients) that are directly applicable
to industrial surface analysis.

*[Add training curves: loss and accuracy per epoch — after training]*

---

### 2.5 Stage 4 — Post-processing (`src/postprocessing.py`) ⬜

*[Complete after zone check implementation]*

Topics to cover:
- ROI (Region of Interest) definition strategy
- Zone boundary check algorithm
- Morphological operations for mask refinement (if segmentation is added)

---

## 3. Experimental Results ⬜

*[Complete after running evaluation — fill tables below]*

### 3.1 Dataset Statistics

| Split | Good samples | Defective samples | Total |
|---|---|---|---|
| Training | — | 0 (MVTec design) | — |
| Test | — | — | — |

Defect types in `metal_nut` test set: *[list from notebook 01]*

### 3.2 Classification Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| HOG + SVM (baseline) | — | — | — | — |
| EfficientNet-B0 (Phase 1) | — | — | — | — |
| EfficientNet-B0 (Phase 2) | — | — | — | — |

### 3.3 Confusion Matrices

*[Insert figures from outputs/results/ after evaluation]*

### 3.4 Training Dynamics

*[Insert loss/accuracy curves after training]*

---

## 4. Failure Analysis ⬜

*[Complete after analyzing misclassified samples]*

Template to fill:

### 4.1 Common Failure Modes

| Failure type | Example | Root cause | Possible mitigation |
|---|---|---|---|
| False negative (missed defect) | *image* | Small defect area < HOG cell size | Reduce pixels_per_cell |
| False positive (good flagged) | *image* | Reflection artifact resembles scratch | Lighting normalization |

### 4.2 Model Limitations

- *[What kinds of defects does the model systematically miss?]*
- *[Does performance degrade for certain part orientations?]*
- *[How does the model behave on out-of-distribution inputs?]*

---

## 5. Ethical Considerations ⬜

*[Complete before final submission]*

### 5.1 Potential Biases

- **Material bias**: the model is trained exclusively on `metal_nut`. Applying
  it to different part geometries or materials without retraining may produce
  unreliable results with no warning to the operator.
- **Lighting bias**: if the training images were captured under a specific
  lighting setup, the model may fail under different factory lighting
  conditions.

### 5.2 Worker Privacy

If cameras are installed above workstations, workers may inadvertently appear
in captured frames. Mitigation strategies:
- Crop images to the part zone only (enforced by the ROI stage).
- Explicit consent and data minimization policy.
- No storage of frames containing identifiable individuals.

### 5.3 Automation and Labor

Automated inspection systems can reduce the need for manual quality control
operators. This raises questions about workforce impact that must be addressed
transparently with all stakeholders before deployment.

### 5.4 False Negative Risk

A false negative (defective part classified as good) has direct safety
implications if the produced component is used in critical systems. The system
should be deployed as an assistant to human inspectors, not as a replacement,
until the false negative rate is quantified and accepted by domain experts.

---

## Appendix — Decision Log

> Record every non-trivial technical decision here as you make it.
> This becomes the backbone of the Methodology section and prepares
> you for the oral examination.

| Date | Decision | Options considered | Reason for choice |
|---|---|---|---|
| 2026-05-05 | Backbone: EfficientNet-B0 | ResNet50, VGG16, EfficientNet-B0 | Best accuracy/parameter tradeoff for small datasets |
| 2026-05-05 | Dataset category: metal_nut | bottle, cable, metal_nut, screw | Most representative of cutting machine parts |
| 2026-05-05 | Target size: 224×224 | 128×128, 224×224, 384×384 | Standard ImageNet size; no architectural changes needed |
| — | *add decisions as you make them* | | |
