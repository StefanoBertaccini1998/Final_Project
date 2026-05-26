"""
main.py — End-to-End Pipeline Entry Point
==========================================
Responsibility: orchestrate all four pipeline stages for a single-image
inference request submitted via the command line.

Pipeline stages executed in sequence:
  Stage 1  preprocessing.preprocess()            Load -> resize -> normalize
  Stage 2  features.extract_hog()                HOG feature extraction (classical path)
           models/deep.get_transforms()           ImageNet transforms (deep path)
  Stage 3  models/classical.ClassicalClassifier   SVM inference (classical path)
           models/deep.DeepClassifier             EfficientNet inference (deep path)
  Stage 4  postprocessing.check_roi()             Spatial position validation

Usage
-----
  # HOG+SVM — trains on category data, then classifies the image:
  python src/main.py --image path/to/image.png --category metal_nut

  # HOG+SVM with saved checkpoint (skips retraining):
  python src/main.py --image path/to/image.png --category metal_nut \\
      --checkpoint outputs/checkpoints/svm_metal_nut.pkl

  # EfficientNet — requires a checkpoint saved by notebook 05:
  python src/main.py --image path/to/image.png --category metal_nut \\
      --model efficientnet \\
      --checkpoint outputs/checkpoints/efficientnet_metal_nut.pt

  # Save HOG+SVM after training for reuse:
  python src/main.py --image path/to/image.png --category metal_nut \\
      --save-checkpoint outputs/checkpoints/svm_metal_nut.pkl
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on sys.path so 'src.*' imports resolve
# regardless of whether this script is invoked as 'python src/main.py'
# or 'python -m src.main'.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import preprocess
from src.features import extract_hog
from src.models.classical import ClassicalClassifier
from src.models.deep import DeepClassifier, get_transforms
from src.postprocessing import define_roi, find_part_bbox, check_roi
from src.dataset import MVTecTorchDataset, MVTEC_CATEGORIES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATA_ROOT = Path("data/mvtec_ad")
DEFAULT_CHECKPOINT_DIR = Path("outputs/checkpoints")

# 0.3 was identified in notebook 06 as the sweet spot for EfficientNet on
# metal_nut: recall improves from 57% to 71% at the cost of 13% precision.
# 0.5 is the standard default for HOG+SVM (no evidence that a lower threshold
# helps for the RBF SVM, which is already well-calibrated).
DEFAULT_THRESHOLD_EFFICIENTNET: float = 0.3
DEFAULT_THRESHOLD_HOG: float = 0.5


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _train_hog_svm(data_root: Path, category: str) -> ClassicalClassifier:
    """Train HOG+SVM on all available data for the category.

    Training on the full dataset (not a held-out split) is appropriate here
    because we are building an inference model, not evaluating it.
    Evaluation is the job of the notebooks.

    Args:
        data_root: Path to the mvtec_ad root directory.
        category: MVTec AD category name.

    Returns:
        Fitted ClassicalClassifier.
    """
    paths, labels = MVTecTorchDataset.collect_paths(data_root / category)
    n_good    = labels.count(0)
    n_defect  = labels.count(1)
    print(f"  Training HOG+SVM on {len(paths)} samples "
          f"({n_good} good / {n_defect} defective) ...")

    X = np.array([extract_hog(preprocess(p)) for p in paths])
    y = np.array(labels)
    clf = ClassicalClassifier().fit(X, y)
    print("  Training complete.")
    return clf


def _load_or_train_hog_svm(
    data_root: Path,
    category: str,
    checkpoint: Path | None,
) -> ClassicalClassifier:
    """Load HOG+SVM from checkpoint if available, otherwise train from scratch.

    Args:
        data_root: Path to mvtec_ad root.
        category: MVTec category name.
        checkpoint: Optional path to a .pkl checkpoint.

    Returns:
        Fitted ClassicalClassifier.
    """
    if checkpoint and checkpoint.exists():
        print(f"  Loading HOG+SVM checkpoint: {checkpoint}")
        return ClassicalClassifier().load(str(checkpoint))
    return _train_hog_svm(data_root, category)


def _load_efficientnet(checkpoint: Path, device: torch.device) -> DeepClassifier:
    """Load a trained EfficientNet checkpoint saved by notebook 05.

    Args:
        checkpoint: Path to a .pt state-dict file.
        device: Torch device (CPU or CUDA).

    Returns:
        DeepClassifier in eval mode.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"EfficientNet checkpoint not found: {checkpoint}\n"
            "Run notebooks/05_deep_learning.ipynb first to train and save the model."
        )
    print(f"  Loading EfficientNet checkpoint: {checkpoint}")
    model = DeepClassifier(num_classes=2)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _classify_hog(
    image: np.ndarray,
    clf: ClassicalClassifier,
    threshold: float,
) -> tuple[bool, float]:
    """Run HOG+SVM inference on a single preprocessed image.

    Args:
        image: Preprocessed float32 array (H, W, 3).
        clf: Fitted ClassicalClassifier.
        threshold: Defect probability above which the part is flagged.

    Returns:
        Tuple of (is_defective: bool, defect_prob: float).
    """
    feat = extract_hog(image).reshape(1, -1)
    prob = float(clf.predict_proba(feat)[0, 1])
    return prob >= threshold, prob


def _classify_efficientnet(
    image: np.ndarray,
    model: DeepClassifier,
    device: torch.device,
    threshold: float,
) -> tuple[bool, float]:
    """Run EfficientNet inference on a single preprocessed image.

    Args:
        image: Preprocessed float32 array (H, W, 3).
        model: DeepClassifier in eval mode.
        device: Torch device.
        threshold: Defect probability above which the part is flagged.

    Returns:
        Tuple of (is_defective: bool, defect_prob: float).
    """
    img_u8  = (image * 255).astype(np.uint8)
    tensor  = get_transforms(train=False)(img_u8).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = float(torch.softmax(model(tensor), dim=1)[0, 1].item())

    return prob >= threshold, prob


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart Factory Vision Monitor — single-image defect inspection."
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the image to inspect."
    )
    parser.add_argument(
        "--category", default="metal_nut", choices=MVTEC_CATEGORIES,
        help="MVTec AD category. Used to locate training data for HOG+SVM. "
             "Default: metal_nut."
    )
    parser.add_argument(
        "--data-root", default=str(DEFAULT_DATA_ROOT),
        help=f"Path to MVTec AD root directory. Default: {DEFAULT_DATA_ROOT}."
    )
    parser.add_argument(
        "--model", default="hog", choices=["hog", "efficientnet"],
        help="Classifier to use. 'hog' trains HOG+SVM on category data (fast). "
             "'efficientnet' loads a saved checkpoint. Default: hog."
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Defect probability threshold. "
             f"Default: {DEFAULT_THRESHOLD_HOG} for HOG, "
             f"{DEFAULT_THRESHOLD_EFFICIENTNET} for EfficientNet."
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to a saved model checkpoint "
             "(.pkl for HOG+SVM, .pt for EfficientNet). "
             "HOG: loads instead of retraining. "
             "EfficientNet: required (use notebook 05 to generate)."
    )
    parser.add_argument(
        "--save-checkpoint", default=None,
        help="Save the trained HOG+SVM model to this path after training. "
             "Ignored for EfficientNet (use the notebook workflow instead)."
    )
    parser.add_argument(
        "--roi-margin", type=float, default=0.20,
        help="ROI margin as a fraction of image size (each side). "
             "Default: 0.20 (designed for wide-field factory cameras). "
             "Use 0.05 for MVTec close-up images where the part fills the frame."
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    data_root  = Path(args.data_root)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.threshold is not None:
        threshold = args.threshold
    elif args.model == "efficientnet":
        threshold = DEFAULT_THRESHOLD_EFFICIENTNET
    else:
        threshold = DEFAULT_THRESHOLD_HOG

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    elif args.model == "hog":
        checkpoint = DEFAULT_CHECKPOINT_DIR / f"svm_{args.category}.pkl"
    else:
        checkpoint = DEFAULT_CHECKPOINT_DIR / f"efficientnet_{args.category}.pt"

    # ------------------------------------------------------------------
    print("=" * 60)
    print("Smart Factory Vision Monitor")
    print("=" * 60)
    print(f"Image    : {image_path}")
    print(f"Category : {args.category}")
    print(f"Model    : {args.model.upper()}  |  Threshold: {threshold:.2f}")
    print(f"Device   : {device}")
    print()

    # Stage 1 — Preprocessing
    print("[Stage 1] Preprocessing ...")
    image = preprocess(str(image_path))
    print(f"  Shape: {image.shape}  dtype: {image.dtype}  "
          f"range: [{image.min():.2f}, {image.max():.2f}]")

    # Stage 2-3 — Feature extraction + Classification
    print(f"\n[Stage 2-3] Classification ({args.model.upper()}) ...")
    if args.model == "hog":
        clf = _load_or_train_hog_svm(data_root, args.category, checkpoint)
        if args.save_checkpoint:
            save_path = Path(args.save_checkpoint)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            clf.save(str(save_path))
            print(f"  Checkpoint saved: {save_path}")
        is_defective, defect_prob = _classify_hog(image, clf, threshold)
    else:
        model = _load_efficientnet(checkpoint, device)
        is_defective, defect_prob = _classify_efficientnet(
            image, model, device, threshold
        )

    verdict_cls = "DEFECTIVE" if is_defective else "GOOD"
    print(f"  Defect probability: {defect_prob:.1%}  -->  {verdict_cls}")

    # Stage 4 — ROI zone check
    print("\n[Stage 4] ROI zone check ...")
    roi  = define_roi(image.shape, margin_pct=args.roi_margin)
    bbox = find_part_bbox(image)
    in_roi, overlap = check_roi(bbox, roi) if bbox else (False, 0.0)
    print(f"  Part bbox : {bbox}")
    print(f"  ROI       : {roi}")
    print(f"  Overlap   : {overlap:.1%}  |  In ROI: {in_roi}")

    # Final verdict
    print()
    print("=" * 60)
    if not in_roi:
        print("VERDICT: REJECT  (reason: part out of position)")
    elif is_defective:
        print(f"VERDICT: REJECT  (reason: defective — prob={defect_prob:.1%})")
    else:
        print(f"VERDICT: PASS    (good part, defect prob={defect_prob:.1%})")
    print("=" * 60)


if __name__ == "__main__":
    main()
