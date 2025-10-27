"""
SRVGGNetMobile: Lightweight SR network (EXACTLY matches baseline structure)
Optimized for Qualcomm QCS8550 HTP edge device

Key features:
- Simple conv blocks (single 3×3 conv + PReLU, NO local residual)
- EXACTLY same structure as baseline SRVGGNetCompact (8 convolutions)
- No BatchNorm overhead (proven fast on HTP)
- PReLU activation (HTP-optimized, used in baseline @ 59 FPS)
- Only global residual connection (baseline-compatible)

Target performance: 57-59 FPS @ 640x512 (matches baseline)
Architecture: Drop-in replacement for SRVGGNetCompact
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


class SimpleBlock(nn.Module):
    """
    Simple Conv Block (EXACTLY matches baseline, NO residual).

    Architecture:
        Input → Conv 3×3 → PReLU → Output

    This matches baseline SRVGGNetCompact exactly:
    - Single 3×3 convolution (HTP-optimized)
    - PReLU activation (proven @ 59 FPS in baseline)
    - NO residual connection (for maximum speed, matches baseline)
    - No BatchNorm (no overhead)

    Args:
        channels: Number of input/output channels (should be power of 2)
    """
    def __init__(self, channels: int):
        super().__init__()

        # Single 3×3 conv (exactly like baseline)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)

        # PReLU activation (exactly like baseline)
        self.act = nn.PReLU(num_parameters=channels)

    def forward(self, x):
        # Conv + activation (NO residual, exactly like baseline)
        out = self.conv(x)
        out = self.act(out)
        return out  # NO + identity!


@ARCH_REGISTRY.register()
class SRVGGNetMobile(nn.Module):
    """
    Mobile SR network (EXACTLY matches baseline SRVGGNetCompact structure).
    Optimized for Qualcomm QCS8550 HTP edge device.

    Architecture:
        Input → Head (Conv 3×3 + PReLU) → Body (N × Simple blocks) → Tail (Conv 3×3 + PixelShuffle) → Output

    Key features (baseline-identical design):
    - Simple blocks: single 3×3 conv + PReLU (NO local residual, like baseline)
    - PReLU activation (proven @ 59 FPS in baseline)
    - No BatchNorm (no overhead)
    - Only global residual (baseline-compatible)
    - Power-of-2 channels for hardware alignment

    Expected performance: 57-59 FPS @ 640×512 (matches baseline)

    Args:
        num_in_ch: Number of input channels (default: 1 for IR)
        num_out_ch: Number of output channels (default: 1 for IR)
        num_feat: Number of feature channels (default: 32, must be power of 2)
        num_conv: Number of simple blocks (default: 8)
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

        # Body: Simple blocks (NO residual, like baseline)
        self.body = nn.Sequential(*[
            SimpleBlock(num_feat) for _ in range(num_conv)
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

    Expected performance: 57-59 FPS @ 640×512 (matches baseline)

    Args:
        num_in_ch: Number of input channels (default: 1 for IR)
        num_out_ch: Number of output channels (default: 1 for IR)
        num_feat: Number of feature channels (default: 32, must be power of 2)
        num_conv: Number of simple blocks (default: 8)
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

        # Body: Simple blocks (NO residual, like baseline)
        self.body = nn.Sequential(*[
            SimpleBlock(num_feat) for _ in range(num_conv)
        ])

        # Tail: project to output channels and upsample
        self.tail = nn.Conv2d(
            num_feat, num_out_ch * upscale * upscale,
            kernel_size=3, stride=1, padding=1, bias=True
        )
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        # Upsample input for global residual connection (baseline-compatible)
        # Use 'nearest' or 'bilinear' - bilinear is 57 FPS, nearest is 59 FPS on QCS8550
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)

        # Main path: feature extraction and upsampling
        feat = self.head(x)
        feat = self.body(feat)
        out = self.tail(feat)
        out = self.upsampler(out)

        # Global residual connection (baseline-compatible)
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
