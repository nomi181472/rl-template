"""Network config dataclasses. No torch dependency."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MLPConfig:
    """Configuration for MLP-based networks."""
    name:              str        = "mlp"
    hidden_sizes:      List[int]  = field(default_factory=lambda: [256, 256])
    activation:        str        = "relu"
    output_activation: Optional[str] = None
    layer_norm:        bool       = False
    dropout:           float      = 0.0
    init_std:          float      = 0.01


@dataclass
class CNNConfig:
    """Configuration for CNN-based networks."""
    name:             str        = "cnn"
    channels:         List[int]  = field(default_factory=lambda: [32, 64, 64])
    kernel_sizes:     List[int]  = field(default_factory=lambda: [8, 4, 3])
    strides:          List[int]  = field(default_factory=lambda: [4, 2, 1])
    mlp_hidden:       List[int]  = field(default_factory=lambda: [512])
    activation:       str        = "relu"
    layer_norm:       bool       = False
    dropout:          float      = 0.0
    init_std:         float      = 0.01
    normalise_pixels: bool       = True
