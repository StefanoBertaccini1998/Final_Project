"""
models/deep.py — Pipeline Stage 3 (Deep Learning): Fine-tuned CNN Classifier
=============================================================================
Responsibility: provide a deep learning classifier that leverages a
pretrained backbone and adapts it to the industrial defect detection task
through transfer learning.

Architecture decision: EfficientNet-B0 as default backbone
-----------------------------------------------------------
EfficientNet-B0 was chosen over ResNet50 and VGG for the following reasons:

  1. Parameter efficiency: EfficientNet-B0 achieves comparable accuracy to
     ResNet50 with ~5.3M parameters vs ~25M, reducing overfitting risk on
     MVTec's relatively small dataset (~300–600 training images per category).

  2. Compound scaling: EfficientNet uniformly scales depth, width, and
     resolution, which empirically yields better feature representations for
     fine-grained texture discrimination — critical for defect detection.

  3. Transfer quality: EfficientNet-B0 pretrained on ImageNet has been shown
     to transfer well to industrial inspection tasks in published literature
     (Bergmann et al., MVTec AD benchmark, 2019).

Training strategy: two-phase fine-tuning
-----------------------------------------
  Phase 1 (freeze_backbone=True): Only the custom head is trained.
    - Rationale: the pretrained backbone already extracts meaningful low-level
      features (edges, textures). Training only the head first avoids
      catastrophic forgetting and converges faster (~5 epochs).
  Phase 2 (unfreeze_backbone()): The entire network is fine-tuned at a
    lower learning rate (1e-5 recommended).
    - Rationale: after the head has stabilized, allowing gradients to flow
      through the backbone adapts the feature extractor to industrial textures
      that differ from ImageNet's natural image distribution.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms


# ---------------------------------------------------------------------------
# ImageNet normalization constants
# Mean and std computed over the ImageNet training set (per channel, RGB).
# All pretrained torchvision models expect input normalized with these values.
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(train: bool = True) -> transforms.Compose:
    """Return the torchvision transform pipeline for a given split.

    Training transforms include geometric and color augmentations that are
    applied on-the-fly on the GPU-bound DataLoader workers, avoiding the
    need to store augmented copies on disk.

    Augmentation choices for training:
      - RandomHorizontalFlip: valid for symmetric industrial parts.
      - RandomRotation(15°): accounts for slight camera misalignment.
      - ColorJitter (brightness/contrast): simulates lighting variation
        across factory shifts without distorting defect geometry.

    Validation/test transforms apply only deterministic operations to ensure
    reproducible metrics.

    Args:
        train: If True, includes data augmentation. If False, only
               resize + normalize (deterministic).

    Returns:
        A torchvision Compose transform ready for use in a DataLoader.
    """
    base = [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
    ]

    if train:
        augmentation = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    else:
        augmentation = []

    normalization = [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    return transforms.Compose(base + augmentation + normalization)


class DeepClassifier(nn.Module):
    """Fine-tuned CNN classifier with a custom task-specific head.

    The model is composed of two parts:
      1. backbone: a pretrained feature extractor (EfficientNet-B0 by default),
         which maps an input image to a dense feature vector.
      2. head: a lightweight MLP that maps the feature vector to class logits.

    The separation between backbone and head enables the two-phase training
    strategy described in the module docstring.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "efficientnet_b0",
        freeze_backbone: bool = True,
        dropout: float = 0.4,
    ):
        """
        Args:
            num_classes: Number of output classes (2 for binary good/defective,
                         N for multi-class defect type classification).
            backbone: Architecture identifier. Supported: 'efficientnet_b0',
                      'resnet50'. EfficientNet-B0 is the default (see module
                      docstring for rationale).
            freeze_backbone: If True, backbone parameters are frozen during
                             Phase 1 training. Set to False or call
                             unfreeze_backbone() to enable Phase 2.
            dropout: Dropout rate in the custom head. 0.4 was selected as a
                     balance between regularization and representational
                     capacity for a binary task with ~500 training samples.
        """
        super().__init__()

        self.backbone_name = backbone
        self.num_classes = num_classes

        # --- Backbone initialization ---
        if backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT
            base = models.efficientnet_b0(weights=weights)
            # in_features: size of the feature vector output by the backbone.
            # For EfficientNet-B0 this is 1280 (from the final AdaptiveAvgPool).
            in_features = base.classifier[1].in_features
            # Remove the original ImageNet head (1000-class FC layer).
            # nn.Identity() is a no-op placeholder that preserves the forward pass.
            base.classifier = nn.Identity()
            self.backbone = base

        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            base = models.resnet50(weights=weights)
            # ResNet50 outputs 2048-dimensional feature vectors from its avgpool.
            in_features = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base

        else:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                "Supported options: 'efficientnet_b0', 'resnet50'."
            )

        # --- Freeze backbone for Phase 1 ---
        if freeze_backbone:
            self._set_backbone_grad(requires_grad=False)

        # --- Custom classification head ---
        # Architecture: Linear(in → 256) → ReLU → Dropout → Linear(256 → num_classes)
        #
        # Why 256 hidden units?
        #   Sufficient expressiveness to learn task-specific patterns from
        #   the backbone's feature vector while small enough to avoid
        #   overfitting on a ~500-sample dataset.
        #
        # Why Dropout before the final layer?
        #   Prevents co-adaptation of neurons in the final mapping. Applied
        #   after the non-linearity so it acts on the activated features.
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through backbone + head.

        Args:
            x: Float tensor of shape (B, 3, 224, 224), normalized with
               ImageNet mean/std (see get_transforms()).

        Returns:
            Logits tensor of shape (B, num_classes). Apply softmax or
            sigmoid externally depending on the loss function.
        """
        features = self.backbone(x)   # (B, in_features)
        logits   = self.head(features) # (B, num_classes)
        return logits

    # -----------------------------------------------------------------------
    # Training phase control
    # -----------------------------------------------------------------------

    def unfreeze_backbone(self) -> None:
        """Unlock backbone parameters for Phase 2 fine-tuning.

        Call this after Phase 1 training has converged (typically after
        ~5–10 epochs). Use a lower learning rate (1e-5) for the optimizer
        to avoid destroying pretrained representations.
        """
        self._set_backbone_grad(requires_grad=True)

    def freeze_backbone(self) -> None:
        """Re-freeze backbone parameters (reverts to Phase 1 mode)."""
        self._set_backbone_grad(requires_grad=False)

    def _set_backbone_grad(self, requires_grad: bool) -> None:
        """Set requires_grad for all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = requires_grad

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def count_parameters(self) -> dict:
        """Return trainable vs total parameter counts.

        Useful for logging and for justifying the model's computational cost
        in the technical analysis document.

        Returns:
            Dict with keys 'trainable', 'frozen', 'total'.
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        return {
            "trainable": trainable,
            "frozen":    total - trainable,
            "total":     total,
        }

    def __repr__(self) -> str:
        params = self.count_parameters()
        return (
            f"DeepClassifier(\n"
            f"  backbone={self.backbone_name},\n"
            f"  num_classes={self.num_classes},\n"
            f"  trainable_params={params['trainable']:,},\n"
            f"  frozen_params={params['frozen']:,}\n"
            f")"
        )
