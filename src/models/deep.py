"""
models/deep.py — Deep Learning Model (Core Logic)
Fine-tuned CNN backbone with a custom classification head.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class DeepClassifier(nn.Module):
    """Fine-tuned CNN with custom classification head.

    Uses a pre-trained backbone (ResNet50 by default) with a
    specialized fully-connected head for the target task.
    The backbone weights are partially frozen for efficient fine-tuning.
    """

    def __init__(self, num_classes: int, backbone: str = "resnet50",
                 freeze_backbone: bool = True, dropout: float = 0.5):
        """
        Args:
            num_classes: Number of output classes.
            backbone: Backbone architecture ('resnet50', 'efficientnet_b0').
            freeze_backbone: If True, freeze backbone weights and only train the head.
            dropout: Dropout probability in the custom head.
        """
        super().__init__()

        # Load pre-trained backbone
        if backbone == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove original head
        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Optionally freeze backbone (only train the custom head)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom classification head
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def unfreeze_backbone(self):
        """Unfreeze backbone for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
