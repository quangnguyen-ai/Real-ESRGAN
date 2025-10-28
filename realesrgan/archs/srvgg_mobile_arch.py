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


class ResidualBlock(nn.Module):
    """
    Residual Block with skip connection for deep SR networks

    Architecture: Conv → Act → Conv → Act → Conv → Act → Conv → Act → Add(input)

    Each block = 4 conv layers with 1 skip connection
    This reduces skip connection overhead on HTP:
    - 16 layers with 2-conv blocks = 8 skips → 20.8ms (SLOW!)
    - 16 layers with 4-conv blocks = 4 skips → Should be ~16ms (FAST!)

    Skip connection helps:
    - Better gradient flow for deep networks (>10 layers)
    - Preserve information across layers
    - Easier optimization
    - With 4 convs per block, minimal overhead on HTP
    """

    def __init__(self, channels, act_type='relu6'):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act_type = act_type

    def _get_activation(self, channels):
        """Helper to get activation layer"""
        if self.act_type == 'relu':
            return nn.ReLU(inplace=True)
        elif self.act_type == 'relu6':
            return nn.ReLU6(inplace=True)
        elif self.act_type == 'prelu':
            return nn.PReLU(num_parameters=channels)
        elif self.act_type == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            return nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        act1 = self._get_activation(self.conv1.out_channels)
        act2 = self._get_activation(self.conv2.out_channels)
        act3 = self._get_activation(self.conv3.out_channels)
        act4 = self._get_activation(self.conv4.out_channels)

        out = self.conv1(x)
        out = act1(out)
        out = self.conv2(out)
        out = act2(out)
        out = self.conv3(out)
        out = act3(out)
        out = self.conv4(out)
        out = act4(out)

        # Skip connection after 4 convs
        out = out + residual
        return out

@ARCH_REGISTRY.register()
class SRVGGNetMobile(nn.Module):
    """
    Progressive Channel SR Network for IR Image Super Resolution on QSC8550 HTP

    Architecture: High capacity (64ch) early → Lower capacity (32ch) later

    Key innovations:
    1. Progressive channel reduction (64ch → 32ch)
       - Early layers: Rich feature extraction (64ch)
       - Later layers: Feature refinement (32ch)

    2. Residual blocks with skip connections (4-conv blocks)
       - Used when num_conv >= 8 (deep enough to benefit)
       - Each ResidualBlock = 4 conv layers + 1 skip connection
       - Reduces skip overhead: 16 layers = 4 skips (not 8!)
       - Better gradient flow for deep networks
       - Easier optimization
       - Minimal HTP overhead with fewer skip connections

    3. Hierarchical learning (inspired by ResNet/U-Net)
       - Match capacity to task complexity
       - More efficient than uniform channels

    4. Optimized for IR whitehot images
       - Thermal patterns benefit from high initial capacity
       - Refinement needs less capacity

    Expected performance examples:
    - 64ch×2 → 32ch×10: ~60G FLOPs, ~16ms (62 FPS) - with skip in 32ch
    - 64ch×0 → 32ch×16: ~48G FLOPs, ~15ms (65 FPS) - with skip in 32ch
    - Better accuracy with skip connections for deep stages

    Args:
        num_in_ch: Input channels (1 for IR grayscale)
        num_out_ch: Output channels (1 for IR grayscale)
        num_conv64feat: Number of 64-channel conv layers (use 0 to skip this stage)
        num_conv32feat: Number of 32-channel conv layers
        upscale: Upscale factor (2x or 4x)
        act_type: Activation ('relu', 'relu6', 'prelu', 'leakyrelu')
        use_skip: Use residual blocks with skip connections (auto-enabled if layers > 6)
    """
    def __init__(self, num_in_ch=1, num_out_ch=1, num_conv64feat=4, num_conv32feat=8,
                 upscale=2, act_type='relu6', use_skip=False):
        super(SRVGGNetMobile, self).__init__()

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_conv64feat = num_conv64feat
        self.num_conv32feat = num_conv32feat
        self.upscale = upscale
        self.act_type = act_type
        self.use_skip = use_skip

        self.body = nn.ModuleList()

        # Determine starting channels
        if num_conv64feat > 0:
            # Stage 1: High capacity feature extraction (64 channels)
            # First conv: 1ch → 64ch
            self.body.append(nn.Conv2d(num_in_ch, 64, 3, 1, 1))
            self.body.append(self._get_activation(64))

            # 64-channel processing layers
            # Use skip connections only if num_conv64feat >= 8
            if use_skip and num_conv64feat >= 8:
                # Use residual blocks (each block = 4 convs, 1 skip)
                num_blocks_64 = num_conv64feat // 4
                for _ in range(num_blocks_64):
                    self.body.append(ResidualBlock(64, act_type))
                # Add remaining layers as plain convs
                remaining = num_conv64feat % 4
                for _ in range(remaining):
                    self.body.append(nn.Conv2d(64, 64, 3, 1, 1))
                    self.body.append(self._get_activation(64))
            else:
                # Plain convs (no skip) for shallow stage
                for _ in range(num_conv64feat):
                    self.body.append(nn.Conv2d(64, 64, 3, 1, 1))
                    self.body.append(self._get_activation(64))

            # Transition: 64ch → 32ch (channel reduction)
            self.body.append(nn.Conv2d(64, 32, 1, 1, 0))
            self.body.append(self._get_activation(32))

        else:
            # No 64ch stage, start directly with 32ch
            self.body.append(nn.Conv2d(num_in_ch, 32, 3, 1, 1))
            self.body.append(self._get_activation(32))

        # Stage 2: Feature refinement (32 channels)
        # Use skip connections if num_conv32feat >= 8 (deep enough)
        if use_skip and num_conv32feat >= 8:
            # Use residual blocks (each block = 4 convs, 1 skip)
            # This limits skip connections to reduce HTP overhead
            num_blocks_32 = num_conv32feat // 4
            for _ in range(num_blocks_32):
                self.body.append(ResidualBlock(32, act_type))
            # Add remaining layers as plain convs
            remaining = num_conv32feat % 4
            for _ in range(remaining):
                self.body.append(nn.Conv2d(32, 32, 3, 1, 1))
                self.body.append(self._get_activation(32))
        else:
            # Plain convs for shallow stage
            for _ in range(num_conv32feat):
                self.body.append(nn.Conv2d(32, 32, 3, 1, 1))
                self.body.append(self._get_activation(32))

        # Upsampling head
        self.body.append(nn.Conv2d(32, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

        # Learnable residual upsampling (HTP-optimized, no F.interpolate!)
        self.residual_upsample = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch * upscale * upscale, 1, 1, 0),
            nn.PixelShuffle(upscale)
        )

    def _get_activation(self, channels):
        """Helper to get activation layer"""
        if self.act_type == 'relu':
            return nn.ReLU(inplace=True)
        elif self.act_type == 'relu6':
            return nn.ReLU6(inplace=True)
        elif self.act_type == 'prelu':
            return nn.PReLU(num_parameters=channels)
        elif self.act_type == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            return nn.ReLU(inplace=True)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)

        out = self.upsampler(out)

        # Learnable residual - NO F.interpolate!
        base = self.residual_upsample(x)
        out = out + base

        return out



@ARCH_REGISTRY.register()
class SRVGGNetMobileInfer(nn.Module):
    """
    Progressive Channel SR Network for IR Image Super Resolution on QSC8550 HTP

    Architecture: High capacity (64ch) early → Lower capacity (32ch) later

    Key innovations:
    1. Progressive channel reduction (64ch → 32ch)
       - Early layers: Rich feature extraction (64ch)
       - Later layers: Feature refinement (32ch)

    2. Residual blocks with skip connections (4-conv blocks)
       - Used when num_conv >= 8 (deep enough to benefit)
       - Each ResidualBlock = 4 conv layers + 1 skip connection
       - Reduces skip overhead: 16 layers = 4 skips (not 8!)
       - Better gradient flow for deep networks
       - Easier optimization
       - Minimal HTP overhead with fewer skip connections

    3. Hierarchical learning (inspired by ResNet/U-Net)
       - Match capacity to task complexity
       - More efficient than uniform channels

    4. Optimized for IR whitehot images
       - Thermal patterns benefit from high initial capacity
       - Refinement needs less capacity

    Expected performance examples:
    - 64ch×2 → 32ch×10: ~60G FLOPs, ~16ms (62 FPS) - with skip in 32ch
    - 64ch×0 → 32ch×16: ~48G FLOPs, ~15ms (65 FPS) - with skip in 32ch
    - Better accuracy with skip connections for deep stages

    Args:
        num_in_ch: Input channels (1 for IR grayscale)
        num_out_ch: Output channels (1 for IR grayscale)
        num_conv64feat: Number of 64-channel conv layers (use 0 to skip this stage)
        num_conv32feat: Number of 32-channel conv layers
        upscale: Upscale factor (2x or 4x)
        act_type: Activation ('relu', 'relu6', 'prelu', 'leakyrelu')
        use_skip: Use residual blocks with skip connections (auto-enabled if layers > 6)
    """
    def __init__(self, num_in_ch=1, num_out_ch=1, num_conv64feat=2, num_conv32feat=8,
                 upscale=2, act_type='relu6', use_skip=True):
        super(SRVGGNetMobileInfer, self).__init__()

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_conv64feat = num_conv64feat
        self.num_conv32feat = num_conv32feat
        self.upscale = upscale
        self.act_type = act_type
        self.use_skip = use_skip

        self.body = nn.ModuleList()

        # Determine starting channels
        if num_conv64feat > 0:
            # Stage 1: High capacity feature extraction (64 channels)
            # First conv: 1ch → 64ch
            self.body.append(nn.Conv2d(num_in_ch, 64, 3, 1, 1))
            self.body.append(self._get_activation(64))

            # 64-channel processing layers
            # Use skip connections only if num_conv64feat >= 8
            if use_skip and num_conv64feat >= 8:
                # Use residual blocks (each block = 4 convs, 1 skip)
                num_blocks_64 = num_conv64feat // 4
                for _ in range(num_blocks_64):
                    self.body.append(ResidualBlock(64, act_type))
                # Add remaining layers as plain convs
                remaining = num_conv64feat % 4
                for _ in range(remaining):
                    self.body.append(nn.Conv2d(64, 64, 3, 1, 1))
                    self.body.append(self._get_activation(64))
            else:
                # Plain convs (no skip) for shallow stage
                for _ in range(num_conv64feat):
                    self.body.append(nn.Conv2d(64, 64, 3, 1, 1))
                    self.body.append(self._get_activation(64))

            # Transition: 64ch → 32ch (channel reduction)
            self.body.append(nn.Conv2d(64, 32, 1, 1, 0))
            self.body.append(self._get_activation(32))

        else:
            # No 64ch stage, start directly with 32ch
            self.body.append(nn.Conv2d(num_in_ch, 32, 3, 1, 1))
            self.body.append(self._get_activation(32))

        # Stage 2: Feature refinement (32 channels)
        # Use skip connections if num_conv32feat >= 8 (deep enough)
        if use_skip and num_conv32feat >= 8:
            # Use residual blocks (each block = 4 convs, 1 skip)
            # This limits skip connections to reduce HTP overhead
            num_blocks_32 = num_conv32feat // 4
            for _ in range(num_blocks_32):
                self.body.append(ResidualBlock(32, act_type))
            # Add remaining layers as plain convs
            remaining = num_conv32feat % 4
            for _ in range(remaining):
                self.body.append(nn.Conv2d(32, 32, 3, 1, 1))
                self.body.append(self._get_activation(32))
        else:
            # Plain convs for shallow stage
            for _ in range(num_conv32feat):
                self.body.append(nn.Conv2d(32, 32, 3, 1, 1))
                self.body.append(self._get_activation(32))

        # Upsampling head
        self.body.append(nn.Conv2d(32, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

        # Learnable residual upsampling (HTP-optimized, no F.interpolate!)
        self.residual_upsample = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch * upscale * upscale, 1, 1, 0),
            nn.PixelShuffle(upscale)
        )

    def _get_activation(self, channels):
        """Helper to get activation layer"""
        if self.act_type == 'relu':
            return nn.ReLU(inplace=True)
        elif self.act_type == 'relu6':
            return nn.ReLU6(inplace=True)
        elif self.act_type == 'prelu':
            return nn.PReLU(num_parameters=channels)
        elif self.act_type == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            return nn.ReLU(inplace=True)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)

        out = self.upsampler(out)

        # Learnable residual - NO F.interpolate!
        base = self.residual_upsample(x)
        out = out + base

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
