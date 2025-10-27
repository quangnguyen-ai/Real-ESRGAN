"""
SRVGGNetMobile: Lightweight SR network with YOLO-optimized Bottleneck Blocks
Optimized for Qualcomm QCS8550 HTP edge device

Key features:
- YOLO-style bottleneck blocks (2 × 3×3 conv with residual)
- BatchNorm + SiLU fusion (HTP-optimized, proven in YOLOv8 @ 250 FPS)
- No 1×1 convolutions (avoids HTP bottleneck)
- Power-of-2 channels for hardware alignment

Target performance: 45-60 FPS @ 640x512, PSNR 27.5-28
Architecture: Inspired by YOLOv8's efficient blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


class YOLOBottleneck(nn.Module):
    """
    YOLO-style Bottleneck block optimized for Qualcomm HTP.

    Architecture:
        Input → Conv 3×3 + BN + SiLU → Conv 3×3 + BN + SiLU → + residual

    This design is proven efficient on QCS8550 (YOLOv8 achieves 250 FPS):
    - Uses only 3×3 convolutions (HTP-optimized)
    - BatchNorm + SiLU fusion by HTP compiler
    - No 1×1 convs (which are slow on HTP)
    - Residual connection for stable training

    Args:
        channels: Number of input/output channels (should be power of 2)
        use_bn: Whether to use BatchNorm (default: True for inference optimization)
    """
    def __init__(self, channels: int, use_bn: bool = True):
        super().__init__()

        # First 3×3 conv
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(channels) if use_bn else nn.Identity()

        # Second 3×3 conv
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=not use_bn)
        self.bn2 = nn.BatchNorm2d(channels) if use_bn else nn.Identity()

        # Activation (SiLU is fused with BN by HTP)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        # Residual connection
        return out + identity


@ARCH_REGISTRY.register()
class SRVGGNetMobile(nn.Module):
    """
    Mobile SR network with YOLO-style Bottleneck blocks.
    Optimized for Qualcomm QCS8550 HTP edge device.

    Architecture:
        Input → Head (Conv 3×3 + BN + SiLU) → Body (N × YOLO Bottleneck) → Tail (Conv 3×3 + PixelShuffle) → Output

    Key features (proven in YOLOv8 @ 250 FPS on same hardware):
    - YOLO bottleneck: 2 × 3×3 conv blocks (HTP-optimized)
    - BatchNorm + SiLU fusion by HTP compiler
    - No 1×1 convolutions (avoids HTP bottleneck)
    - Power-of-2 channels for hardware alignment

    Args:
        num_in_ch: Number of input channels (default: 1 for IR)
        num_out_ch: Number of output channels (default: 1 for IR)
        num_feat: Number of feature channels (default: 32, must be power of 2)
        num_conv: Number of YOLO bottleneck blocks (default: 8)
        upscale: Upscale factor (default: 2)
        use_bn: Use BatchNorm for HTP fusion (default: True)
    """
    def __init__(self,
                 num_in_ch: int = 1,
                 num_out_ch: int = 1,
                 num_feat: int = 32,
                 num_conv: int = 8,
                 upscale: int = 2,
                 use_bn: bool = True,
                 # Legacy parameters for backward compatibility (ignored)
                 expansion: float = None,
                 attention_freq: int = None,
                 act_type: str = None):
        super().__init__()

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.use_bn = use_bn

        # Head: expand to feature channels with BN + SiLU
        self.head = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1, bias=not use_bn),
            nn.BatchNorm2d(num_feat) if use_bn else nn.Identity(),
            nn.SiLU(inplace=True)
        )

        # Body: YOLO bottleneck blocks
        self.body = nn.Sequential(*[
            YOLOBottleneck(num_feat, use_bn=use_bn) for _ in range(num_conv)
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
        num_conv: Number of YOLO bottleneck blocks (default: 8)
        upscale: Upscale factor (default: 2)
        use_bn: Use BatchNorm for HTP fusion (default: True)
    """
    def __init__(self,
                 num_in_ch: int = 1,
                 num_out_ch: int = 1,
                 num_feat: int = 32,
                 num_conv: int = 8,
                 upscale: int = 2,
                 use_bn: bool = True,
                 # Legacy parameters for backward compatibility (ignored)
                 expansion: float = None,
                 attention_freq: int = None,
                 act_type: str = None):
        super().__init__()

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.use_bn = use_bn

        # Head: expand to feature channels with BN + SiLU
        self.head = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1, bias=not use_bn),
            nn.BatchNorm2d(num_feat) if use_bn else nn.Identity(),
            nn.SiLU(inplace=True)
        )

        # Body: YOLO bottleneck blocks
        self.body = nn.Sequential(*[
            YOLOBottleneck(num_feat, use_bn=use_bn) for _ in range(num_conv)
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
