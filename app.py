"""Gradio demo — Smart Factory Vision Monitor.

Comparative interface: HOG+SVM vs EfficientNet-B0 on MVTec AD images.
Upload an image or click a thumbnail from the built-in gallery to classify.
Grad-CAM is generated on demand (separate button) to keep inference fast.
"""

import os
import sys
import logging
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(__file__))

import numpy as np
import gradio as gr
import torch
import cv2
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------------------------
# Logging — outputs to stdout so Railway captures it in the dashboard
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# Log environment at startup
import sklearn, skimage, joblib as _joblib
log.info("=== Smart Factory Vision Monitor starting ===")
log.info("Python      : %s", sys.version.split()[0])
log.info("sklearn     : %s", sklearn.__version__)
log.info("skimage     : %s", skimage.__version__)
log.info("numpy       : %s", np.__version__)
log.info("joblib      : %s", _joblib.__version__)
log.info("torch       : %s", torch.__version__)
log.info("cv2         : %s", cv2.__version__)

from src.preprocessing import preprocess
from src.models.classical import ClassicalClassifier
from src.models.deep import DeepClassifier, get_transforms
from src.postprocessing import define_roi, find_part_bbox, check_roi, draw_overlay

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICE = torch.device("cpu")  # Railway has no GPU
CKPT_DIR = Path("outputs/checkpoints")
EXAMPLES_DIR = Path("examples")

CATEGORIES = ["metal_nut", "carpet", "leather", "wood", "tile", "grid", "cable"]

# HOG+SVM checkpoint available only for metal_nut
HOG_CATEGORIES = ["metal_nut"]

# metal_nut V2 was trained with weighted loss — optimal threshold is 0.5.
# Texture-category models (notebook 09) have no weighted loss fix, so they
# retain the class-imbalance bias toward PASS. t=0.3 corrects this, matching
# the threshold used when F1 was measured in notebook 09.
ENET_THRESHOLD_BY_CAT: dict[str, float] = {
    "metal_nut": 0.5,
    "carpet":    0.3,
    "leather":   0.3,
    "wood":      0.3,
    "tile":      0.3,
    "grid":      0.3,
    "cable":     0.3,
}
LOW_CONF_LOW  = 0.25
LOW_CONF_HIGH = 0.55

GRADCAM_ALPHA = 0.5  # image/heatmap blend — 0.5 gives equal weight to both

# ---------------------------------------------------------------------------
# Model cache — loaded once at startup
# ---------------------------------------------------------------------------
_hog_models: dict[str, ClassicalClassifier] = {}
_enet_models: dict[str, DeepClassifier] = {}
_gradcam_objects: dict[str, "GradCAM"] = {}  # one GradCAM per category, reused across calls


def _load_hog(category: str) -> ClassicalClassifier | None:
    if category in _hog_models:
        return _hog_models[category]
    ckpt = CKPT_DIR / f"svm_{category}.pkl"
    if not ckpt.exists():
        log.warning("HOG checkpoint not found: %s", ckpt)
        return None
    clf = ClassicalClassifier()
    clf.load(str(ckpt))
    _hog_models[category] = clf
    log.info("HOG model loaded: %s | classes=%s | n_support=%s",
             ckpt.name, clf.model.classes_.tolist(), clf.model.n_support_.tolist())
    return clf


def _load_enet(category: str) -> DeepClassifier | None:
    if category in _enet_models:
        return _enet_models[category]
    ckpt = CKPT_DIR / f"efficientnet_{category}.pt"
    if not ckpt.exists():
        log.warning("EfficientNet checkpoint not found: %s", ckpt)
        return None
    model = DeepClassifier(num_classes=2)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.unfreeze_backbone()   # required for GradCAM backward hooks
    model.eval()
    _enet_models[category] = model
    log.info("EfficientNet loaded: %s | device=%s | gradcam_available=%s",
             ckpt.name, DEVICE, GRADCAM_AVAILABLE)
    # Build GradCAM once and reuse — creating it per-call adds ~70s on CPU
    if GRADCAM_AVAILABLE:
        from pytorch_grad_cam import GradCAM as _GradCAM
        target_layers = [model.backbone.features[-1][0]]
        _gradcam_objects[category] = _GradCAM(model=model, target_layers=target_layers)
    return model


# Models are loaded on first use (lazy) to keep startup fast on Railway.


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
_BANNER_BASE = (
    "border-radius:6px;padding:10px 14px;margin-bottom:8px;"
    "font-size:14px;font-weight:500;"
)

def _error_html(msg: str) -> str:
    return (
        f"<div style='{_BANNER_BASE}background:#f8d7da;border:1px solid #f5c2c7;color:#842029'>"
        f"❌ <b>Error</b>: {msg}</div>"
    )

def _warn_html(msg: str) -> str:
    return (
        f"<div style='{_BANNER_BASE}background:#fff3cd;border:1px solid #ffc107;color:#664d03'>"
        f"⚠️ {msg}</div>"
    )

def _info_html(msg: str) -> str:
    return (
        f"<div style='{_BANNER_BASE}background:#cff4fc;border:1px solid #9eeaf9;color:#055160'>"
        f"ℹ️ {msg}</div>"
    )


def _hog_classify(image_float: np.ndarray, category: str) -> tuple[str, float] | None:
    """Return (verdict, prob) for HOG+SVM, or None if no checkpoint."""
    try:
        clf = _load_hog(category)
        if clf is None:
            return None
        from src.features import extract_hog
        feat = extract_hog(image_float).reshape(1, -1)
        full_proba = clf.predict_proba(feat)
        prob = float(full_proba[0, 1])
        verdict = "REJECT" if prob >= 0.5 else "PASS"
        log.info("HOG [%s] feat_mean=%.4f feat_std=%.4f proba=%s classes=%s -> %s",
                 category, float(feat.mean()), float(feat.std()),
                 full_proba[0].round(3).tolist(), clf.model.classes_.tolist(), verdict)
        return verdict, prob
    except Exception as e:
        log.error("HOG inference failed [%s]: %s", category, e, exc_info=True)
        return "ERROR", 0.0


def _enet_classify_fast(image_float: np.ndarray, category: str) -> tuple[str, float]:
    """Forward pass only — no GradCAM. Returns (verdict, prob)."""
    try:
        model = _load_enet(category)
        if model is None:
            return "N/A", 0.0
        img_u8 = (image_float * 255).astype(np.uint8)
        tensor = get_transforms(train=False)(img_u8).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = float(torch.softmax(model(tensor), dim=1)[0, 1].item())
        threshold = ENET_THRESHOLD_BY_CAT.get(category, 0.5)
        verdict = "REJECT" if prob >= threshold else "PASS"
        log.info("ENet [%s] prob=%.4f threshold=%.2f -> %s", category, prob, threshold, verdict)
        return verdict, prob
    except Exception as e:
        log.error("EfficientNet inference failed [%s]: %s", category, e, exc_info=True)
        return "ERROR", 0.0


def _enet_gradcam(image_float: np.ndarray, category: str) -> np.ndarray | None:
    """Grad-CAM backward pass only. Returns RGB overlay or None."""
    try:
        if not GRADCAM_AVAILABLE or category not in _gradcam_objects:
            log.warning("GradCAM unavailable for [%s] (available=%s)", category, GRADCAM_AVAILABLE)
            return None
        model = _load_enet(category)
        if model is None:
            return None
        img_u8 = (image_float * 255).astype(np.uint8)
        tensor = get_transforms(train=False)(img_u8).unsqueeze(0).to(DEVICE)
        cam = _gradcam_objects[category]
        with torch.enable_grad():
            heatmap = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(1)])[0]
        log.info("GradCAM [%s] heatmap min=%.3f max=%.3f", category, float(heatmap.min()), float(heatmap.max()))
        return show_cam_on_image(image_float.astype(np.float32), heatmap, use_rgb=True, image_weight=0.5)
    except Exception as e:
        log.error("GradCAM failed [%s]: %s", category, e, exc_info=True)
        return None


def _roi_overlay(image_float: np.ndarray) -> np.ndarray:
    """Return ROI zone-check overlay image."""
    try:
        roi = define_roi(image_float.shape, margin_pct=0.05)
        part_bbox = find_part_bbox(image_float)
        in_roi, overlap = check_roi(part_bbox, roi) if part_bbox else (True, 1.0)
        img_rgb = (image_float * 255).astype(np.uint8)
        result = draw_overlay(img_rgb, roi, part_bbox, in_roi, is_defective=False)
        log.info("ROI: part_bbox=%s in_roi=%s overlap=%.2f", part_bbox, in_roi, overlap)
        return result
    except Exception as e:
        log.warning("ROI overlay failed: %s", e)
        return (image_float * 255).astype(np.uint8)


def _preprocess_image(image: np.ndarray) -> np.ndarray:
    """Save to temp file and run preprocess() pipeline (includes denoising).

    Used only for user-uploaded images. Gallery images use _preprocess_path()
    to avoid Gradio thumbnail downscaling corrupting HOG features.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        Image.fromarray(image).convert("RGB").save(tmp_path)
        log.info("preprocess_image: saved array shape=%s to tempfile, calling preprocess()", image.shape)
        return preprocess(tmp_path)   # float32 [0,1] 224×224, denoised
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Gallery helpers
# ---------------------------------------------------------------------------
def get_gallery_images(category: str) -> list[tuple[str, str]]:
    """Return list of (path, label) for the selected category."""
    cat_dir = EXAMPLES_DIR / category
    if not cat_dir.exists():
        return []
    images = []
    for p in sorted(cat_dir.glob("*.png")):
        # filename: bent_000.png → label: bent (defective) / good_000.png → good
        defect = p.stem.rsplit("_", 1)[0]
        label = "GOOD" if defect == "good" else f"DEFECT: {defect}"
        images.append((str(p), label))
    return images


# ---------------------------------------------------------------------------
# Main classify function (fast — no Grad-CAM)
# ---------------------------------------------------------------------------
def classify(image: np.ndarray | None, category: str, gallery_path: str | None = None) -> tuple:
    """Run HOG+SVM and EfficientNet (forward only). Returns results + stored image_float.

    gallery_path: original on-disk path of the selected gallery image.
    When set, preprocess() is called directly on that file, bypassing the
    numpy→tempfile→preprocess round-trip that Gradio thumbnail downscaling corrupts.
    """
    empty = np.zeros((224, 224, 3), dtype=np.uint8)

    if image is None:
        return (
            empty, "—", "—",       # roi, hog_verdict, hog_prob
            empty, "—", "—", "",   # gradcam_placeholder, enet_verdict, enet_prob, warning
            None,                  # image_float state (no image to store)
        )

    try:
        if gallery_path and Path(gallery_path).exists():
            log.info("classify: using original gallery file %s", gallery_path)
            image_float = preprocess(gallery_path)
        else:
            log.info("classify: uploaded image array shape=%s", image.shape)
            image_float = _preprocess_image(image)

        # ROI check
        roi_img = _roi_overlay(image_float)

        # HOG + SVM
        hog_result = _hog_classify(image_float, category)
        if hog_result is None:
            hog_verdict = "N/A"
            hog_prob_str = "No HOG model for this category"
        elif hog_result[0] == "ERROR":
            hog_verdict = "ERROR"
            hog_prob_str = "Inference failed — see logs"
        else:
            hog_verdict, hog_prob = hog_result
            hog_prob_str = f"{hog_prob:.1%}"

        # EfficientNet — forward pass only (fast)
        enet_verdict, enet_prob = _enet_classify_fast(image_float, category)
        threshold = ENET_THRESHOLD_BY_CAT.get(category, 0.5)
        if enet_verdict == "ERROR":
            enet_prob_str = "Inference failed — see logs"
            warning_html = _error_html("EfficientNet inference failed. Check Railway logs.")
        else:
            enet_prob_str = f"{enet_prob:.1%}  (threshold {threshold:.0%})"
            warning_html = ""
            if LOW_CONF_LOW <= enet_prob <= LOW_CONF_HIGH:
                warning_html = _warn_html(
                    f"<b>Low confidence</b> ({enet_prob:.0%}) — result may be unreliable "
                    "on out-of-distribution images."
                )

        # Grad-CAM placeholder: show preprocessed image until user requests heatmap
        gradcam_placeholder = (image_float * 255).astype(np.uint8)

        return (
            roi_img, hog_verdict, hog_prob_str,
            gradcam_placeholder, enet_verdict, enet_prob_str, warning_html,
            image_float,
        )

    except Exception as e:
        log.error("classify failed [%s]: %s", category, e, exc_info=True)
        error_html = _error_html(f"Classification failed: {e}")
        return (empty, "ERROR", str(e), empty, "ERROR", str(e), error_html, None)


# ---------------------------------------------------------------------------
# Deferred Grad-CAM function
# ---------------------------------------------------------------------------
def run_gradcam(image_float: np.ndarray | None, category: str) -> np.ndarray:
    """Run Grad-CAM on the last classified image (stored in state)."""
    if image_float is None:
        log.warning("run_gradcam called with no stored image — classify first")
        return np.zeros((224, 224, 3), dtype=np.uint8)
    try:
        overlay = _enet_gradcam(image_float, category)
        if overlay is None:
            log.warning("GradCAM returned None for [%s] — returning plain image", category)
            return (image_float * 255).astype(np.uint8)
        return overlay
    except Exception as e:
        log.error("run_gradcam failed [%s]: %s", category, e, exc_info=True)
        return (image_float * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="Smart Factory Vision Monitor",
    theme=gr.themes.Soft(),
    css="""
        .model-header {
            font-size: 15px; font-weight: 700; padding: 6px 0 10px 0;
            border-bottom: 2px solid #e2e8f0; margin-bottom: 10px;
        }
        footer { display:none !important; }
    """,
) as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML(
        "<div style='background:linear-gradient(135deg,#1e293b 0%,#0f172a 100%);"
        "border-radius:10px;padding:20px 24px;margin-bottom:12px;'>"
        "<h1 style='color:#f1f5f9;margin:0 0 4px 0;font-size:22px;font-weight:700;"
        "letter-spacing:-0.3px;'>Smart Factory Vision Monitor</h1>"
        "<p style='color:#94a3b8;margin:0;font-size:13px;'>"
        "Comparative defect detection &middot; HOG+SVM baseline vs EfficientNet-B0 V2 &middot; "
        "MVTec Anomaly Detection dataset &middot; EPICODE Computer Vision Project"
        "</p></div>"
    )

    # ── Model stats cards ───────────────────────────────────────────────────
    gr.HTML(
        "<div style='display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap;'>"

        "<div style='flex:1;min-width:170px;background:#f8fafc;border-radius:8px;"
        "padding:12px 16px;border-left:4px solid #64748b;'>"
        "<div style='font-size:11px;color:#64748b;font-weight:700;"
        "text-transform:uppercase;letter-spacing:.06em;'>HOG + SVM</div>"
        "<div style='font-size:22px;font-weight:800;color:#1e293b;margin:3px 0;'>F1 0.605</div>"
        "<div style='font-size:12px;color:#64748b;'>Acc 83.2% &middot; Recall 46% &middot; Prec 87%</div>"
        "</div>"

        "<div style='flex:1;min-width:170px;background:#f0fdf4;border-radius:8px;"
        "padding:12px 16px;border-left:4px solid #22c55e;'>"
        "<div style='font-size:11px;color:#15803d;font-weight:700;"
        "text-transform:uppercase;letter-spacing:.06em;'>EfficientNet-B0 V2</div>"
        "<div style='font-size:22px;font-weight:800;color:#14532d;margin:3px 0;'>F1 0.792</div>"
        "<div style='font-size:12px;color:#15803d;'>Acc 88.1% &middot; Recall 68% &middot; Prec 95%</div>"
        "</div>"

        "<div style='flex:1;min-width:170px;background:#eff6ff;border-radius:8px;"
        "padding:12px 16px;border-left:4px solid #3b82f6;'>"
        "<div style='font-size:11px;color:#1d4ed8;font-weight:700;"
        "text-transform:uppercase;letter-spacing:.06em;'>Dataset</div>"
        "<div style='font-size:22px;font-weight:800;color:#1e3a8a;margin:3px 0;'>7 categories</div>"
        "<div style='font-size:12px;color:#1d4ed8;'>MVTec AD &middot; metal_nut primary target</div>"
        "</div>"

        "<div style='flex:1;min-width:170px;background:#fdf4ff;border-radius:8px;"
        "padding:12px 16px;border-left:4px solid #a855f7;'>"
        "<div style='font-size:11px;color:#7e22ce;font-weight:700;"
        "text-transform:uppercase;letter-spacing:.06em;'>Improvement V1&rarr;V2</div>"
        "<div style='font-size:22px;font-weight:800;color:#581c87;margin:3px 0;'>+22% F1</div>"
        "<div style='font-size:12px;color:#7e22ce;'>Weighted loss &middot; FN: 15&rarr;9</div>"
        "</div>"

        "</div>"
    )

    # Hidden state: preprocessed float image for deferred GradCAM
    image_float_state = gr.State(None)
    # Hidden state: original on-disk path of the last gallery selection
    # (bypasses Gradio thumbnail downscaling that corrupts HOG features)
    gallery_path_state = gr.State(None)

    with gr.Row():
        # ── Left column: input controls ────────────────────────────────────
        with gr.Column(scale=1):
            gr.HTML(
                "<div style='font-size:11px;font-weight:700;color:#64748b;"
                "text-transform:uppercase;letter-spacing:.06em;"
                "margin-bottom:6px;'>Input</div>"
            )
            category_dd = gr.Dropdown(
                choices=CATEGORIES,
                value="metal_nut",
                label="Category",
            )
            gallery = gr.Gallery(
                value=get_gallery_images("metal_nut"),
                label="Example images — click to load",
                columns=4,
                height=210,
                object_fit="cover",
                allow_preview=False,
            )
            image_input = gr.Image(
                type="numpy",
                label="Selected / uploaded image",
                height=250,
            )
            run_btn = gr.Button("Classify", variant="primary", size="lg")
            gr.HTML(
                "<div style='font-size:11px;color:#94a3b8;margin-top:6px;line-height:1.6;'>"
                "HOG+SVM available for <b>metal_nut</b> only.<br>"
                "Other categories: EfficientNet-B0 only."
                "</div>"
            )

        # ── Right column: results ───────────────────────────────────────────
        with gr.Column(scale=2):
            warning_out = gr.HTML()

            with gr.Row():
                # ── HOG + SVM ──────────────────────────────────────────────
                with gr.Column():
                    gr.HTML(
                        "<div class='model-header' style='color:#374151;'>"
                        "HOG + SVM"
                        "<span style='font-size:11px;font-weight:400;color:#9ca3af;"
                        "margin-left:8px;'>classical baseline</span></div>"
                    )
                    hog_verdict_out  = gr.Label(label="Verdict")
                    hog_prob_out     = gr.Textbox(
                        label="Defect probability", interactive=False
                    )
                    roi_out          = gr.Image(
                        label="ROI zone check", height=210
                    )

                # ── EfficientNet-B0 ────────────────────────────────────────
                with gr.Column():
                    gr.HTML(
                        "<div class='model-header' style='color:#15803d;'>"
                        "EfficientNet-B0 V2"
                        "<span style='font-size:11px;font-weight:400;color:#9ca3af;"
                        "margin-left:8px;'>weighted loss &middot; threshold shown below</span></div>"
                    )
                    enet_verdict_out = gr.Label(label="Verdict")
                    enet_prob_out    = gr.Textbox(
                        label="Defect probability", interactive=False
                    )
                    gradcam_out      = gr.Image(
                        label="Grad-CAM heatmap", height=210
                    )
                    gradcam_btn      = gr.Button(
                        "Generate Grad-CAM  (slow on CPU — ~30 s)",
                        variant="secondary",
                    )

    # ── Footer metrics table ────────────────────────────────────────────────
    gr.HTML(
        "<div style='margin-top:18px;padding:14px 18px;background:#f8fafc;"
        "border-radius:8px;border:1px solid #e2e8f0;'>"
        "<div style='font-size:11px;font-weight:700;color:#64748b;"
        "text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;'>"
        "metal_nut benchmark &mdash; full results"
        "</div>"
        "<table style='width:100%;border-collapse:collapse;font-size:12px;'>"
        "<thead><tr style='background:#e2e8f0;border-bottom:2px solid #cbd5e1;'>"
        "<th style='text-align:left;padding:6px 10px;color:#374151;font-weight:700;'>Model</th>"
        "<th style='padding:6px 10px;color:#374151;font-weight:700;'>Accuracy</th>"
        "<th style='padding:6px 10px;color:#374151;font-weight:700;'>F1</th>"
        "<th style='padding:6px 10px;color:#374151;font-weight:700;'>Recall</th>"
        "<th style='padding:6px 10px;color:#374151;font-weight:700;'>Precision</th>"
        "<th style='padding:6px 10px;color:#374151;font-weight:700;'>FN / 28</th>"
        "</tr></thead>"
        "<tbody>"
        "<tr style='background:#f1f5f9;border-bottom:1px solid #e2e8f0;'>"
        "<td style='padding:6px 10px;font-weight:600;color:#1e293b;'>HOG + SVM</td>"
        "<td style='text-align:center;padding:6px 10px;color:#1e293b;'>83.2%</td>"
        "<td style='text-align:center;padding:6px 10px;color:#1e293b;'>0.605</td>"
        "<td style='text-align:center;padding:6px 10px;color:#1e293b;'>46.4%</td>"
        "<td style='text-align:center;padding:6px 10px;color:#1e293b;'>86.7%</td>"
        "<td style='text-align:center;padding:6px 10px;color:#1e293b;'>15</td>"
        "</tr>"
        "<tr style='background:#f8fafc;border-bottom:1px solid #e2e8f0;'>"
        "<td style='padding:6px 10px;font-weight:600;color:#1e293b;'>EfficientNet V1 (t=0.5)</td>"
        "<td style='text-align:center;padding:6px 10px;color:#1e293b;'>85.1%</td>"
        "<td style='text-align:center;padding:6px 10px;color:#1e293b;'>0.634</td>"
        "<td style='text-align:center;padding:6px 10px;color:#1e293b;'>46.4%</td>"
        "<td style='text-align:center;padding:6px 10px;color:#1e293b;'>100%</td>"
        "<td style='text-align:center;padding:6px 10px;color:#1e293b;'>15</td>"
        "</tr>"
        "<tr style='background:#dcfce7;font-weight:700;border-bottom:1px solid #bbf7d0;'>"
        "<td style='padding:6px 10px;color:#15803d;'>EfficientNet V2 (t=0.5) &#9733;</td>"
        "<td style='text-align:center;padding:6px 10px;color:#15803d;'>88.1%</td>"
        "<td style='text-align:center;padding:6px 10px;color:#15803d;'>0.792</td>"
        "<td style='text-align:center;padding:6px 10px;color:#15803d;'>67.9%</td>"
        "<td style='text-align:center;padding:6px 10px;color:#15803d;'>95.0%</td>"
        "<td style='text-align:center;padding:6px 10px;color:#15803d;'>9</td>"
        "</tr>"
        "</tbody></table>"
        "<div style='font-size:11px;color:#94a3b8;margin-top:6px;'>"
        "V2 fix: CrossEntropyLoss(weight=[1.0, 2.60]) + Phase 2 extended to 20 epochs. "
        "Grad-CAM pointing accuracy 4.3% &mdash; model attends to global shape, not local defect region."
        "</div>"
        "</div>"
    )

    # ── Events ─────────────────────────────────────────────────────────────
    def update_gallery(category: str):
        return gr.Gallery(value=get_gallery_images(category))

    def load_from_gallery(evt: gr.SelectData, category: str):
        # Resolve original on-disk path via evt.index — avoids Gradio's
        # thumbnail cache, which downsizes images and corrupts HOG features.
        originals = get_gallery_images(category)
        original_path = originals[evt.index][0] if evt.index < len(originals) else None
        log.info("gallery_select: index=%d original_path=%s", evt.index, original_path)
        src = original_path or (
            evt.value["image"]["path"] if isinstance(evt.value, dict) else evt.value
        )
        return np.array(Image.open(src).convert("RGB")), original_path

    def clear_gallery_path():
        return None

    category_dd.change(fn=update_gallery, inputs=category_dd, outputs=gallery)
    gallery.select(
        fn=load_from_gallery,
        inputs=[category_dd],
        outputs=[image_input, gallery_path_state],
    )
    # Clear the stored gallery path when the user uploads their own image
    image_input.upload(fn=clear_gallery_path, outputs=[gallery_path_state])

    run_btn.click(
        fn=classify,
        inputs=[image_input, category_dd, gallery_path_state],
        outputs=[roi_out, hog_verdict_out, hog_prob_out,
                 gradcam_out, enet_verdict_out, enet_prob_out, warning_out,
                 image_float_state],
    )

    gradcam_btn.click(
        fn=run_gradcam,
        inputs=[image_float_state, category_dd],
        outputs=[gradcam_out],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
