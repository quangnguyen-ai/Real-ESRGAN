"""
SRVGGNetMobile: Lightweight SR network with Simple Residual Blocks
Optimized for Qualcomm QCS8550 HTP edge device

Key features:
- Simple residual blocks (single 3×3 conv + PReLU + residual)
- Same computation as baseline SRVGGNetCompact (8 convolutions)
- No BatchNorm overhead (proven fast on HTP)
- PReLU activation (HTP-optimized, used in baseline @ 60 FPS)
- Residual connections for stable training and better gradient flow

Target performance: 55-60 FPS @ 640x512, PSNR 27.5-28
Architecture: Baseline-compatible with residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


class SimpleResidualBlock(nn.Module):
    """
    Simple Residual Block optimized for Qualcomm HTP.

    Architecture:
        Input → Conv 3×3 → PReLU → + residual → Output

    This design is baseline-compatible with residual connections:
    - Single 3×3 convolution (HTP-optimized, like baseline)
    - PReLU activation (proven fast, used in baseline @ 60 FPS)
    - No BatchNorm (no overhead)
    - Residual connection for stable training

    Args:
        channels: Number of input/output channels (should be power of 2)
    """
    def __init__(self, channels: int):
        super().__init__()

        # Single 3×3 conv (like baseline)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)

        # PReLU activation (baseline-compatible)
        # self.act = nn.PReLU(num_parameters=channels)
        self.act = nn.ReLU6(inplace=True)
    def forward(self, x):
        identity = x

        # Conv + activation
        out = self.conv(x)
        out = self.act(out)

        # Residual connection
        return out + identity


@ARCH_REGISTRY.register()
class SRVGGNetMobile(nn.Module):
    """
    Mobile SR network with Simple Residual blocks.
    Optimized for Qualcomm QCS8550 HTP edge device.

    Architecture:
        Input → Head (Conv 3×3 + PReLU) → Body (N × Simple Residual blocks) → Tail (Conv 3×3 + PixelShuffle) → Output

    Key features (baseline-compatible design):
    - Simple residual: single 3×3 conv per block (same as baseline)
    - PReLU activation (proven @ 60 FPS in baseline)
    - No BatchNorm (no overhead)
    - Residual connections (better training stability)
    - Power-of-2 channels for hardware alignment

    Args:
        num_in_ch: Number of input channels (default: 1 for IR)
        num_out_ch: Number of output channels (default: 1 for IR)
        num_feat: Number of feature channels (default: 32, must be power of 2)
        num_conv: Number of simple residual blocks (default: 8)
        upscale: Upscale factor (default: 2)
    """
    def __init__(self,
                 num_in_ch: int = 1,
                 num_out_ch: int = 1,
                 num_feat: int = 32,
                 num_conv: int = 8,
                 upscale: int = 2,
                 # Legacy parameters for backward compatibility (ignored)
                 use_bn: bool = None,
                 expansion: float = None,
                 attention_freq: int = None,
                 act_type: str = None):
        super().__init__()

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale

        # Head: expand to feature channels with PReLU (baseline-compatible)
        self.head = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU(num_parameters=num_feat)
        )

        # Body: Simple residual blocks
        self.body = nn.Sequential(*[
            SimpleResidualBlock(num_feat) for _ in range(num_conv)
        ])

        # Tail: project to output channels and upsample
        self.tail = nn.Conv2d(
            num_feat, num_out_ch * upscale * upscale,
            kernel_size=3, stride=1, padding=1, bias=True
        )
        self.upsampler = nn.PixelShuffle(upscale)

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
        num_feat: Number of feature channels (default: 32, must be power of 2)
        num_conv: Number of simple residual blocks (default: 8)
        upscale: Upscale factor (default: 2)
    """
    def __init__(self,
                 num_in_ch: int = 1,
                 num_out_ch: int = 1,
                 num_feat: int = 32,
                 num_conv: int = 8,
                 upscale: int = 2,
                 # Legacy parameters for backward compatibility (ignored)
                 use_bn: bool = None,
                 expansion: float = None,
                 attention_freq: int = None,
                 act_type: str = None):
        super().__init__()

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale

        # Head: expand to feature channels with PReLU (baseline-compatible)
        self.head = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.PReLU(num_parameters=num_feat)
            # nn.SiLU(inplace=True)
             nn.ReLU6(inplace=True)
        )

        # Body: Simple residual blocks
        self.body = nn.Sequential(*[
            SimpleResidualBlock(num_feat) for _ in range(num_conv)
        ])

        # Tail: project to output channels and upsample
        self.tail = nn.Conv2d(
            num_feat, num_out_ch * upscale * upscale,
            kernel_size=3, stride=1, padding=1, bias=True
        )
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        # Upsample input for global residual connection
        base = F.interpolate(
            x, scale_factor=self.upscale,
            mode='bilinear', align_corners=False
        )
        # base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        # Main path: feature extraction and upsampling
        feat = self.head(x)
        feat = self.body(feat)
        out = self.tail(feat)
        out = self.upsampler(out)

        # VI
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
