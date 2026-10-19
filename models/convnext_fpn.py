import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.transformer_attention import TransformerAttention


class ConvNeXtFPNTransformer(nn.Module):

    def __init__(
        self,
        fpn_channels=256,
        num_heads=8,
        num_transformer_layers=2
    ):
        super().__init__()

        # --------------------------------------------------
        # ConvNeXt backbone
        # --------------------------------------------------

        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=True,
            features_only=True
        )

        backbone_channels = (
            self.backbone.feature_info.channels()
        )

        # --------------------------------------------------
        # FPN lateral projections
        # --------------------------------------------------

        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(
                channels,
                fpn_channels,
                kernel_size=1
            )
            for channels in backbone_channels
        ])

        # --------------------------------------------------
        # Shared multi-scale feature fusion
        # --------------------------------------------------

        self.fusion = nn.Sequential(
            nn.Conv2d(
                fpn_channels * len(backbone_channels),
                fpn_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )

        # --------------------------------------------------
        # Transformer self-attention
        # --------------------------------------------------

        self.transformer = TransformerAttention(
            input_channels=fpn_channels,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dropout=0.1
        )

        self.output_channels = fpn_channels

    def forward(self, x):

        # --------------------------------------------------
        # 1. ConvNeXt feature extraction
        # --------------------------------------------------

        features = self.backbone(x)

        # --------------------------------------------------
        # 2. Convert all scales to common channel size
        # --------------------------------------------------

        fpn_features = []

        # Use the third feature-map resolution as the
        # common spatial resolution.
        target_size = features[2].shape[-2:]

        for feature, lateral in zip(
            features,
            self.lateral_layers
        ):

            feature = lateral(feature)

            feature = F.interpolate(
                feature,
                size=target_size,
                mode="bilinear",
                align_corners=False
            )

            fpn_features.append(feature)

        # --------------------------------------------------
        # 3. Shared Feature Fusion
        # --------------------------------------------------

        fused_features = torch.cat(
            fpn_features,
            dim=1
        )

        fused_features = self.fusion(
            fused_features
        )

        # --------------------------------------------------
        # 4. Self-Attention
        # --------------------------------------------------

        attention_features = self.transformer(
            fused_features
        )

        return attention_features


if __name__ == "__main__":

    print("=" * 60)
    print("CONVNEXT-FPN + TRANSFORMER FEATURE EXTRACTOR TEST")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    model = ConvNeXtFPNTransformer()

    model = model.to(device)

    model.eval()

    dummy_input = torch.randn(
        2,
        3,
        224,
        224
    ).to(device)

    with torch.no_grad():

        output = model(
            dummy_input
        )

    print()
    print("Input shape: ", dummy_input.shape)
    print("Output shape:", output.shape)
    print()
    print(
        "Output channels:",
        model.output_channels
    )

    print()
    print("=" * 60)
    print("FEATURE EXTRACTOR TEST COMPLETED")
    print("=" * 60)