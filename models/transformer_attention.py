import torch
import torch.nn as nn


class TransformerAttention(nn.Module):

    def __init__(
        self,
        input_channels=256,
        num_heads=8,
        num_layers=2,
        dropout=0.1
    ):

        super().__init__()

        self.input_channels = input_channels

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_channels,
            nhead=num_heads,
            dim_feedforward=input_channels * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Final normalization
        self.norm = nn.LayerNorm(
            input_channels
        )

    def forward(self, x):

        # Expected input:
        # [Batch, Channels, Height, Width]

        batch_size, channels, height, width = x.shape

        # Convert spatial feature map into tokens
        # [B, C, H, W]
        #       ↓
        # [B, H*W, C]

        tokens = x.flatten(
            start_dim=2
        ).transpose(
            1, 2
        )

        # Self-attention through Transformer
        tokens = self.transformer(
            tokens
        )

        # Normalize transformer features
        tokens = self.norm(
            tokens
        )

        # Convert tokens back to feature map
        # [B, H*W, C]
        #       ↓
        # [B, C, H, W]

        output = tokens.transpose(
            1, 2
        ).reshape(
            batch_size,
            channels,
            height,
            width
        )

        return output


if __name__ == "__main__":

    print("=" * 60)
    print("PALM RIPENESS PREDICTION")
    print("TRANSFORMER SELF-ATTENTION TEST")
    print("=" * 60)

    # Create sample feature map
    sample = torch.randn(
        2,
        256,
        14,
        14
    )

    print()
    print("Input shape:")
    print(sample.shape)

    # Create Transformer
    model = TransformerAttention(
        input_channels=256,
        num_heads=8,
        num_layers=2
    )

    # Forward pass
    output = model(sample)

    print()
    print("Output shape:")
    print(output.shape)

    print()
    print("=" * 60)
    print("TRANSFORMER MODULE CREATED SUCCESSFULLY")
    print("=" * 60)