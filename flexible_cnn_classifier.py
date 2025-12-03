import torch
import torch.nn as nn


class FlexibleCNN(nn.Module):

    def __init__(
        self,
        in_channels=3,
        num_conv_blocks=2,
        base_filters=16,
        fc_hidden_dim=64,
        num_classes=10,
        dropout_rate=0.25,
        seed=42,
    ):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            num_conv_blocks: Number of convolutional blocks (2, 3, or 4)
            base_filters: Starting number of filters, doubles each block
            fc_hidden_dim: Hidden dimension size in fully connected layer
            num_classes: Number of output classes
            dropout_rate: Dropout probability
            seed: Random seed for reproducibility
        """
        super(FlexibleCNN, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.in_channels = in_channels
        self.num_conv_blocks = num_conv_blocks
        self.base_filters = base_filters
        self.fc_hidden_dim = fc_hidden_dim
        self.num_classes = num_classes

        # build convolutional blocks dynamically
        conv_layers = []
        in_ch = in_channels
        out_ch = base_filters

        for block_idx in range(num_conv_blocks):
            # conv -> ReLU -> MaxPool
            conv_layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            in_ch = out_ch
            out_ch = out_ch * 2  # Double filters each block

        self.convolutional_layers = nn.Sequential(*conv_layers)

        # calc flattened size after convolutions
        # for cifar (32x32): after n blocks, size = 32 / (2^n)
        self.final_conv_channels = base_filters * (2 ** (num_conv_blocks - 1))

        # calc spatial dimensions after pooling, cifar is 32
        spatial_size = 32 // (2**num_conv_blocks)

        self.flattened_size = self.final_conv_channels * spatial_size * spatial_size

        # FC layers
        self.fully_connected_layers = nn.Sequential(
            nn.Linear(self.flattened_size, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = torch.flatten(x, 1)
        x = self.fully_connected_layers(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_architecture_summary(self):
        return {
            "in_channels": self.in_channels,
            "num_conv_blocks": self.num_conv_blocks,
            "base_filters": self.base_filters,
            "fc_hidden_dim": self.fc_hidden_dim,
            "num_classes": self.num_classes,
            "final_conv_channels": self.final_conv_channels,
            "flattened_size": self.flattened_size,
            "total_parameters": self.count_parameters(),
        }


def create_baseline_model(in_channels=3, num_classes=10, seed=42):
    return FlexibleCNN(
        in_channels=in_channels,
        num_conv_blocks=2,
        base_filters=16,
        fc_hidden_dim=64,
        num_classes=num_classes,
        seed=seed,
    )


def create_medium_model(in_channels=3, num_classes=10, seed=42):
    return FlexibleCNN(
        in_channels=in_channels,
        num_conv_blocks=3,
        base_filters=32,
        fc_hidden_dim=128,
        num_classes=num_classes,
        seed=seed,
    )


def create_high_model(in_channels=3, num_classes=10, seed=42):
    return FlexibleCNN(
        in_channels=in_channels,
        num_conv_blocks=4,
        base_filters=64,
        fc_hidden_dim=256,
        num_classes=num_classes,
        seed=seed,
    )


def create_very_high_model(in_channels=3, num_classes=10, seed=42):
    return FlexibleCNN(
        in_channels=in_channels,
        num_conv_blocks=4,
        base_filters=128,
        fc_hidden_dim=512,
        num_classes=num_classes,
        seed=seed,
    )


def create_extreme_model(in_channels=3, num_classes=10, seed=42):
    """
    Even higher-capacity model for probing the overparameterized regime.
    """
    return FlexibleCNN(
        in_channels=in_channels,
        num_conv_blocks=4,
        base_filters=256,
        fc_hidden_dim=1024,
        num_classes=num_classes,
        seed=seed,
    )


def create_width_scaled_model(width_multiplier, in_channels=3, num_classes=10, seed=42):
    """
    Create model with width scaled by multiplier (like ResNet18 width in the paper).

    This allows fine-grained control over model capacity for observing double descent.
    Width multiplier of 1 = baseline, higher = more parameters.

    Key for double descent:
    - NO dropout (dropout_rate=0.0) to not smooth out the interpolation peak
    - Width scales base_filters and fc_hidden_dim
    """
    base_filters = max(
        1, int(8 * width_multiplier)
    )  # Start small to hit interpolation threshold
    fc_hidden_dim = max(8, int(32 * width_multiplier))

    return FlexibleCNN(
        in_channels=in_channels,
        num_conv_blocks=2,  # Keep architecture simple
        base_filters=base_filters,
        fc_hidden_dim=fc_hidden_dim,
        num_classes=num_classes,
        dropout_rate=0.0,  # NO DROPOUT - critical for seeing double descent peak!
        seed=seed,
    )


def get_width_multipliers_for_double_descent():
    """
    Returns width multipliers that span the interpolation threshold.

    For CIFAR-10 with ~10k training samples (subset) and typical CNN:
    - Small widths (1-5): underparameterized regime
    - Medium widths (6-15): around interpolation threshold (peak error)
    - Large widths (16-128): overparameterized regime (where DD recovery happens)

    Note: Using smaller training subset (10k) makes interpolation threshold 
    easier to hit and the peak more pronounced.
    """
    # Fine-grained sampling around expected interpolation threshold
    # Extended to 128 to ensure we're well into overparameterized regime
    widths = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  # underparameterized
        12, 14, 16, 18, 20,              # around interpolation threshold
        24, 32, 40, 48, 64, 80, 96, 128  # overparameterized
    ]
    return widths


if __name__ == "__main__":
    models = {
        "Baseline": create_baseline_model(),
        "Medium": create_medium_model(),
        "High": create_high_model(),
        "Very High": create_very_high_model(),
        "Extreme": create_extreme_model(),
    }

    for name, model in models.items():
        summary = model.get_architecture_summary()
        print(f"\n{name} model:")
        print(f"  conv blocks: {summary['num_conv_blocks']}")
        print(f"  base filters: {summary['base_filters']}")
        print(f"  FC hidden dim: {summary['fc_hidden_dim']}")
        print(f"  total params: {summary['total_parameters']:,}")

    # test forward pass
    x = torch.randn(4, 3, 32, 32)  # Batch of 4 CIFAR-10 images
    output = models["Baseline"](x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert tuple(output.shape) == (4, 10)
    print("great success!")
