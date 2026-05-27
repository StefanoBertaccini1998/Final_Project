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
from src.postprocessing import define_roi, check_roi, draw_overlay

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

# V2 model (weighted loss) performs best at t=0.5: F1=0.792, Precision=0.950.
# Threshold tuning was a workaround for V1's class-imbalance bias; V2 no longer needs it.
ENET_THRESHOLD = 0.5
LOW_CONF_LOW  = 0.35
LOW_CONF_HIGH = 0.65

GRADCAM_ALPHA = 0.4

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
def _hog_classify(image_float: np.ndarray, category: str) -> tuple[str, float] | None:
    """Return (verdict, prob) for HOG+SVM, or None if no checkpoint."""
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


def _enet_classify_fast(image_float: np.ndarray, category: str) -> tuple[str, float]:
    """Forward pass only — no GradCAM. Returns (verdict, prob)."""
    model = _load_enet(category)
    if model is None:
        return "N/A", 0.0

    img_u8 = (image_float * 255).astype(np.uint8)
    tensor = get_transforms(train=False)(img_u8).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = float(torch.softmax(model(tensor), dim=1)[0, 1].item())

    verdict = "REJECT" if prob >= ENET_THRESHOLD else "PASS"
    return verdict, prob


def _enet_gradcam(image_float: np.ndarray, category: str) -> np.ndarray | None:
    """Grad-CAM backward pass only. Returns RGB overlay or None."""
    if not GRADCAM_AVAILABLE or category not in _gradcam_objects:
        return None

    # Ensure model is loaded (may not be if gradcam called before classify)
    model = _load_enet(category)
    if model is None:
        return None

    img_u8 = (image_float * 255).astype(np.uint8)
    tensor = get_transforms(train=False)(img_u8).unsqueeze(0).to(DEVICE)

    cam = _gradcam_objects[category]
    with torch.enable_grad():
        heatmap = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(1)])[0]
    return show_cam_on_image(image_float.astype(np.float32), heatmap, use_rgb=True, image_weight=1 - GRADCAM_ALPHA)


def _roi_overlay(image_float: np.ndarray) -> np.ndarray:
    """Return ROI zone-check overlay image."""
    try:
        roi = define_roi(image_float, margin=0.05)
        status, overlap = check_roi(image_float, roi)
        img_bgr = cv2.cvtColor((image_float * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        overlay_bgr = draw_overlay(img_bgr, roi, status, overlap)
        return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
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
    else:
        hog_verdict, hog_prob = hog_result
        hog_prob_str = f"{hog_prob:.1%}"

    # EfficientNet — forward pass only (fast)
    enet_verdict, enet_prob = _enet_classify_fast(image_float, category)
    enet_prob_str = f"{enet_prob:.1%}  (threshold {ENET_THRESHOLD:.0%})"

    # Low-confidence warning
    warning_html = ""
    if LOW_CONF_LOW <= enet_prob <= LOW_CONF_HIGH:
        warning_html = (
            "<div style='background:#fff3cd;border:1px solid #ffc107;"
            "border-radius:6px;padding:10px;margin-top:8px;font-size:14px'>"
            "⚠️ <b>Low confidence</b> — result may be unreliable on "
            "out-of-distribution images.</div>"
        )

    # Grad-CAM placeholder: show preprocessed image until user requests heatmap
    gradcam_placeholder = (image_float * 255).astype(np.uint8)

    return (
        roi_img, hog_verdict, hog_prob_str,
        gradcam_placeholder, enet_verdict, enet_prob_str, warning_html,
        image_float,   # stored in gr.State for deferred GradCAM
    )


# ---------------------------------------------------------------------------
# Deferred Grad-CAM function
# ---------------------------------------------------------------------------
def run_gradcam(image_float: np.ndarray | None, category: str) -> np.ndarray:
    """Run Grad-CAM on the last classified image (stored in state)."""
    if image_float is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    overlay = _enet_gradcam(image_float, category)
    if overlay is None:
        return (image_float * 255).astype(np.uint8)
    return overlay


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Smart Factory Vision Monitor", theme=gr.themes.Default()) as demo:

    gr.Markdown(
        "# Smart Factory Vision Monitor\n"
        "**Comparative defect detection**: HOG+SVM classical baseline vs "
        "EfficientNet-B0 deep learning. Select a category, pick an example "
        "or upload your own image."
    )

    # Hidden state: preprocessed float image for deferred GradCAM
    image_float_state = gr.State(None)
    # Hidden state: original on-disk path of the last gallery selection
    # (bypasses Gradio thumbnail downscaling that corrupts HOG features)
    gallery_path_state = gr.State(None)

    with gr.Row():
        # ── Left column: input controls ────────────────────────────────────
        with gr.Column(scale=1):
            category_dd = gr.Dropdown(
                choices=CATEGORIES,
                value="metal_nut",
                label="Category",
            )
            gallery = gr.Gallery(
                value=get_gallery_images("metal_nut"),
                label="Examples (click to load)",
                columns=4,
                height=220,
                object_fit="cover",
                allow_preview=False,
            )
            image_input = gr.Image(
                type="numpy",
                label="Input image",
                height=260,
            )
            run_btn = gr.Button("Classify", variant="primary")

        # ── Right column: results ───────────────────────────────────────────
        with gr.Column(scale=2):
            warning_out = gr.HTML(label="")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### HOG + SVM")
                    hog_verdict_out  = gr.Label(label="Verdict")
                    hog_prob_out     = gr.Textbox(label="Defect probability", interactive=False)
                    roi_out          = gr.Image(label="ROI zone check", height=200)

                with gr.Column():
                    gr.Markdown("### EfficientNet-B0 (V2 — weighted loss)")
                    enet_verdict_out = gr.Label(label="Verdict")
                    enet_prob_out    = gr.Textbox(label="Defect probability", interactive=False)
                    gradcam_out      = gr.Image(label="Grad-CAM heatmap", height=200)
                    gradcam_btn      = gr.Button("Generate Grad-CAM  (slow ~30s on CPU)")

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

    gr.Markdown(
        "---\n"
        "*Metal nut: HOG+SVM available. Other categories: EfficientNet only.*  \n"
        "*EfficientNet threshold: 0.5 (V2 weighted loss). "
        "HOG threshold: 0.5 (default).*"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
