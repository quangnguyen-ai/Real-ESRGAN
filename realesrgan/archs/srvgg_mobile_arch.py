"""
SRVGGNetMobile: Lightweight SR network with Inverted Residual Blocks
Optimized for edge devices (Qualcomm QCS8550 HTP)

Key features:
- Inverted Residual blocks (MobileNetV2 style) for rich feature extraction
- ECA attention for quality boost with minimal overhead
- Optimized for 1-channel IR images on edge devices

Target performance: 60-68 FPS @ 640x512, PSNR ~28
Architecture: narrow→wide→narrow with residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


def make_activation(act_type: str, num_channels: int = None):
    """
    Create activation layer.

    Args:
        act_type: 'relu', 'leakyrelu', 'prelu'
        num_channels: Required for prelu
    """
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif act_type == 'prelu':
        if num_channels is None:
            raise ValueError('num_channels required for prelu')
        return nn.PReLU(num_parameters=num_channels)
    else:
        raise ValueError(f'Unsupported activation: {act_type}')


class ECAAttention(nn.Module):
    """
    Efficient Channel Attention (ECA) - lightweight attention mechanism.

    Only 5 parameters, <1% overhead, but significant quality improvement.
    Paper: ECA-Net (CVPR 2020)

    Args:
        channels: Number of input channels
        k_size: Adaptive kernel size for 1D conv (default: 3)
    """
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global average pooling: [B, C, H, W] -> [B, C, 1, 1]
        y = self.avg_pool(x)

        # 1D convolution along channel dimension: [B, C, 1, 1] -> [B, 1, C] -> [B, 1, C] -> [B, C, 1, 1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Sigmoid activation for channel weights
        y = self.sigmoid(y)

        # Apply attention weights
        return x * y.expand_as(x)


class InvertedResidualBlock(nn.Module):
    """
    Inverted Residual Block (MobileNetV2 style) for super-resolution.

    Architecture:
        Input (narrow) → Expand 1×1 (wide) → DW 3×3 → Project 1×1 (narrow) + residual

    This design allows DW conv to work on richer feature space while keeping
    input/output narrow for efficiency.

    Args:
        num_channels: Number of input/output channels (narrow width)
        expansion: Expansion ratio (typically 2-4)
        act_type: Activation type
        use_attention: Whether to use ECA attention
    """
    def __init__(self,
                 num_channels: int,
                 expansion: float = 2.5,
                 act_type: str = 'prelu',
                 use_attention: bool = False):
        super().__init__()

        hidden_channels = int(num_channels * expansion)

        # Expand: narrow → wide
        self.expand = nn.Sequential(
            nn.Conv2d(num_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True),
            make_activation(act_type, hidden_channels)
        )

        # Depthwise: spatial processing on wide features
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                hidden_channels, hidden_channels,
                kernel_size=3, stride=1, padding=1,
                groups=hidden_channels, bias=True
            ),
            make_activation(act_type, hidden_channels)
        )

        # Project: wide → narrow
        self.project = nn.Conv2d(
            hidden_channels, num_channels,
            kernel_size=1, stride=1, padding=0, bias=True
        )

        # Optional attention
        self.attention = ECAAttention(num_channels) if use_attention else None

    def forward(self, x):
        identity = x

        # Inverted residual path
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)

        # Apply attention if enabled
        if self.attention is not None:
            out = self.attention(out)

        # Residual connection
        return out + identity


@ARCH_REGISTRY.register()
class SRVGGNetMobile(nn.Module):
    """
    Mobile SR network with Inverted Residual blocks (MobileNetV2 style).
    Optimized for Qualcomm QCS8550 HTP edge device.

    Architecture:
        Input → Head (3×3 conv) → Body (N × Inverted Residual blocks) → Tail (3×3 + PixelShuffle) → Output

    Key improvements over pure depthwise separable:
    - Inverted residual: narrow→wide→narrow for richer feature extraction
    - ECA attention: every 4 blocks for quality boost
    - Better receptive field and cross-channel mixing

    Args:
        num_in_ch: Number of input channels (default: 1 for IR)
        num_out_ch: Number of output channels (default: 1 for IR)
        num_feat: Number of feature channels - narrow width (default: 40)
        num_conv: Number of inverted residual blocks (default: 12)
        expansion: Expansion ratio for inverted residual (default: 2.5)
        attention_freq: Add ECA attention every N blocks (0=disable, default: 4)
        upscale: Upscale factor (default: 2)
        act_type: Activation type ('relu', 'leakyrelu', 'prelu')
    """
    def __init__(self,
                 num_in_ch: int = 1,
                 num_out_ch: int = 1,
                 num_feat: int = 40,
                 num_conv: int = 12,
                 expansion: float = 2.5,
                 attention_freq: int = 4,
                 upscale: int = 2,
                 act_type: str = 'prelu'):
        super().__init__()

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.expansion = expansion
        self.attention_freq = attention_freq
        self.upscale = upscale
        self.act_type = act_type

        # Head: expand to feature channels
        self.head = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1, bias=True),
            make_activation(act_type, num_feat)
        )

        # Body: inverted residual blocks with periodic attention
        body = []
        for i in range(num_conv):
            # Add attention every N blocks (if enabled)
            use_attention = (attention_freq > 0) and ((i + 1) % attention_freq == 0)
            body.append(InvertedResidualBlock(
                num_channels=num_feat,
                expansion=expansion,
                act_type=act_type,
                use_attention=use_attention
            ))
        self.body = nn.Sequential(*body)

        # Tail: project to output channels and upsample
        self.tail = nn.Conv2d(
            num_feat, num_out_ch * upscale * upscale,
            kernel_size=3, stride=1, padding=1, bias=True
        )
        self.upsampler = nn.PixelShuffle(upscale)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Kaiming normal for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.PReLU):
                nn.init.constant_(m.weight, 0.25)

    def forward(self, x):
        # Upsample input for global residual connection
        base = F.interpolate(
            x, scale_factor=self.upscale,
            mode='bilinear', align_corners=False
        )

        # Main path: feature extraction and upsampling
        feat = self.head(x)
        feat = self.body(feat)
        out = self.tail(feat)
        out = self.upsampler(out)

        # Global residual connection
        out = out + base

        return out


@ARCH_REGISTRY.register()
class SRVGGNetMobileInfer(nn.Module):
    """
    Inference version of SRVGGNetMobile with output clamping.

    Same as SRVGGNetMobile but clamps output to [0, 1] for inference.
    Use this for deployment/inference to ensure valid pixel values.

    Args:
        num_in_ch: Number of input channels (default: 1 for IR)
        num_out_ch: Number of output channels (default: 1 for IR)
        num_feat: Number of feature channels - narrow width (default: 40)
        num_conv: Number of inverted residual blocks (default: 12)
        expansion: Expansion ratio for inverted residual (default: 2.5)
        attention_freq: Add ECA attention every N blocks (0=disable, default: 4)
        upscale: Upscale factor (default: 2)
        act_type: Activation type ('relu', 'leakyrelu', 'prelu')
    """
    def __init__(self,
                 num_in_ch: int = 1,
                 num_out_ch: int = 1,
                 num_feat: int = 32,
                 num_conv: int = 12,
                 expansion: float = 2.0,
                 attention_freq: int = 0,
                 upscale: int = 2,
                 act_type: str = 'prelu'):
        super().__init__()

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.expansion = expansion
        self.attention_freq = attention_freq
        self.upscale = upscale
        self.act_type = act_type

        # Head: expand to feature channels
        self.head = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1, bias=True),
            make_activation(act_type, num_feat)
        )

        # Body: inverted residual blocks with periodic attention
        body = []
        for i in range(num_conv):
            # Add attention every N blocks (if enabled)
            use_attention = (attention_freq > 0) and ((i + 1) % attention_freq == 0)
            body.append(InvertedResidualBlock(
                num_channels=num_feat,
                expansion=expansion,
                act_type=act_type,
                use_attention=use_attention
            ))
        self.body = nn.Sequential(*body)

        # Tail: project to output channels and upsample
        self.tail = nn.Conv2d(
            num_feat, num_out_ch * upscale * upscale,
            kernel_size=3, stride=1, padding=1, bias=True
        )
        self.upsampler = nn.PixelShuffle(upscale)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Kaiming normal for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.PReLU):
                nn.init.constant_(m.weight, 0.25)

    def forward(self, x):
        # Upsample input for global residual connection
        base = F.interpolate(
            x, scale_factor=self.upscale,
            mode='bilinear', align_corners=False
        )

        # Main path: feature extraction and upsampling
        feat = self.head(x)
        feat = self.body(feat)
        out = self.tail(feat)
        out = self.upsampler(out)

        # Global residual connection
        out = out + base

        # Clamp output to valid range [0, 1] for inference
        out = torch.clamp(out, 0.0, 1.0)

        return out


@ARCH_REGISTRY.register()
class SRVGGNetHybrid(nn.Module):
    """
    Hybrid architecture: standard convs at head/tail, depthwise in body.
    Better quality with slightly more FLOPs than pure mobile version.

    Use this when:
    - Need better quality than SRVGGNetMobile
    - Still want 2-3x speedup vs SRVGGNetCompact

    Args:
        num_in_ch: Number of input channels
        num_out_ch: Number of output channels
        num_feat: Number of feature channels
        num_conv: Number of depthwise blocks
        upscale: Upscale factor
        act_type: Activation type
    """
    def __init__(self,
                 num_in_ch: int = 3,
                 num_out_ch: int = 3,
                 num_feat: int = 32,
                 num_conv: int = 8,
                 upscale: int = 4,
                 act_type: str = 'leakyrelu'):
        super().__init__()

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale

        # Head: 2 standard convs for feature extraction
        self.head = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat // 2, 3, 1, 1, bias=True),
            make_activation(act_type, num_feat // 2),
            nn.Conv2d(num_feat // 2, num_feat, 3, 1, 1, bias=True),
            make_activation(act_type, num_feat)
        )

        # Body: depthwise separable blocks
        body = []
        for _ in range(num_conv):
            body.append(DepthwiseSeparableBlock(num_feat, act_type))
        self.body = nn.Sequential(*body)

        # Pre-tail: 1 standard conv for refinement
        self.pre_tail = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True),
            make_activation(act_type, num_feat)
        )

        # Tail: upsampling
        self.tail = nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1, bias=True)
        self.upsampler = nn.PixelShuffle(upscale)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.PReLU):
                nn.init.constant_(m.weight, 0.25)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)

        feat = self.head(x)
        feat = self.body(feat)
        feat = self.pre_tail(feat)
        out = self.tail(feat)
        out = self.upsampler(out)
        out = out + base

        return out
