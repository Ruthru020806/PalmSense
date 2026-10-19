import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convnext_fpn import ConvNeXtFPNTransformer


class MultiTaskFFBModel(nn.Module):

    def __init__(
        self,
        num_classes=4,
        feature_channels=256
    ):
        super().__init__()

        # ==================================================
        # SHARED BACKBONE
        # ConvNeXt-FPN + Transformer
        # ==================================================

        self.shared_backbone = ConvNeXtFPNTransformer(
            fpn_channels=feature_channels,
            num_heads=8,
            num_transformer_layers=2
        )

        # ==================================================
        # 1. FFB INSTANCE / ROI MASK HEAD
        #
        # Produces a single-channel spatial mask.
        # This is compatible with the existing
        # box-derived pseudo-mask targets.
        # ==================================================

        self.mask_head = nn.Sequential(

            nn.Conv2d(
                feature_channels,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                1,
                kernel_size=1
            )
        )

        # ==================================================
        # 2. RIPENESS CLASSIFICATION HEAD
        #
        # 4 classes:
        # Empty Bunch
        # Overripe
        # Ripe
        # Underripe
        # ==================================================

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.ripeness_head = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                feature_channels,
                256
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                num_classes
            )
        )

        # ==================================================
        # 3. DENSITY / COUNT HEAD
        #
        # Predicts FFB count.
        #
        # Current dataset contains one FFB per image,
        # so this is currently a count regression target.
        # ==================================================

        self.density_head = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                feature_channels,
                128
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                1
            )
        )

    def forward(self, x):

        # ==================================================
        # Shared feature extraction
        # ==================================================

        features = self.shared_backbone(x)

        # ==================================================
        # Segmentation / ROI mask
        # ==================================================

        mask_logits = self.mask_head(
            features
        )

        # Upsample to original image size
        mask_logits = F.interpolate(
            mask_logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # ==================================================
        # Global shared representation
        # ==================================================

        pooled_features = self.global_pool(
            features
        )

        # ==================================================
        # Ripeness classification
        # ==================================================

        ripeness_logits = self.ripeness_head(
            pooled_features
        )

        # ==================================================
        # Density / count regression
        # ==================================================

        density_count = self.density_head(
            pooled_features
        )

        return {
            "mask_logits": mask_logits,
            "ripeness_logits": ripeness_logits,
            "density_count": density_count
        }


if __name__ == "__main__":

    print("=" * 60)
    print("MULTI-TASK FFB MODEL TEST")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    model = MultiTaskFFBModel()

    model = model.to(device)

    model.eval()

    # --------------------------------------------------
    # Dummy input
    # --------------------------------------------------

    dummy_input = torch.randn(
        2,
        3,
        224,
        224
    ).to(device)

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    with torch.no_grad():

        outputs = model(
            dummy_input
        )

    print()
    print("Input:")
    print(dummy_input.shape)

    print()
    print("Multi-task outputs:")

    print(
        "Mask logits:     ",
        outputs["mask_logits"].shape
    )

    print(
        "Ripeness logits: ",
        outputs["ripeness_logits"].shape
    )

    print(
        "Density count:   ",
        outputs["density_count"].shape
    )

    print()
    print("=" * 60)
    print("MULTI-TASK MODEL TEST COMPLETED")
    print("=" * 60)