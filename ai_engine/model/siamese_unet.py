"""Siamese U-Net architecture for disaster building localization and damage classification.

Architecture specification:
- Shared encoder (ResNet34 / ResNet18 via segmentation_models_pytorch, ImageNet weights) processes PRE and POST.
- Multi-scale feature fusion: at each scale, concatenate [pre, post, |post - pre|].
- U-Net decoder with skip connections progressive upsampling to full image resolution.
- Two task heads:
  (a) Building localization head: 1 channel, predicts footprint presence
  (b) Damage classification head: 5 channels (0: bg, 1: no-damage, 2: minor, 3: major, 4: destroyed)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class DecoderBlock(nn.Module):
    """U-Net decoder block: 2x bilinear upsampling, skip concatenation, double conv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            # Handle potential shape mismatch due to odd dimensions
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class SiameseUNet(nn.Module):
    """Siamese U-Net for two-stage satellite disaster triage."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        num_classes: int = 5,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32),
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.num_classes = num_classes

        # Shared pre/post optical encoder
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights,
        )

        enc_channels = self.encoder.out_channels  # e.g., (3, 64, 64, 128, 256, 512) for resnet34

        # Concatenation of [pre, post, |post - pre|] triples each stage's channel count
        # Stage 5 (deepest bottleneck)
        fused_channels_5 = enc_channels[5] * 3
        self.bottleneck = nn.Sequential(
            nn.Conv2d(fused_channels_5, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Decoder blocks
        # Block 1: 512 -> 256 (skip from stage 4)
        skip4 = enc_channels[4] * 3
        self.decoder_block1 = DecoderBlock(512, skip4, decoder_channels[0])

        # Block 2: 256 -> 128 (skip from stage 3)
        skip3 = enc_channels[3] * 3
        self.decoder_block2 = DecoderBlock(decoder_channels[0], skip3, decoder_channels[1])

        # Block 3: 128 -> 64 (skip from stage 2)
        skip2 = enc_channels[2] * 3
        self.decoder_block3 = DecoderBlock(decoder_channels[1], skip2, decoder_channels[2])

        # Block 4: 64 -> 32 (skip from stage 1)
        skip1 = enc_channels[1] * 3
        self.decoder_block4 = DecoderBlock(decoder_channels[2], skip1, decoder_channels[3])

        # Final projection to full image resolution
        self.final_up = nn.Sequential(
            nn.Conv2d(decoder_channels[3], 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Head 1: Building localization (1 channel)
        self.loc_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        # Head 2: Damage classification (num_classes channels, 5 by default)
        self.dmg_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )

    def forward(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through shared Siamese encoder and dual-head decoder.

        Args:
            pre: Pre-disaster image tensor of shape (B, 3, H, W)
            post: Post-disaster image tensor of shape (B, 3, H, W)

        Returns:
            (loc_logits, dmg_logits):
            - loc_logits: Tensor of shape (B, 1, H, W)
            - dmg_logits: Tensor of shape (B, 5, H, W)
        """
        # 1. Shared encoder feature extraction
        feats_pre = self.encoder(pre)
        feats_post = self.encoder(post)

        # 2. Multi-scale feature difference and fusion
        fused = []
        for i in range(1, len(feats_pre)):
            fp = feats_pre[i]
            fq = feats_post[i]
            diff = torch.abs(fq - fp)
            fused.append(torch.cat([fp, fq, diff], dim=1))

        # fused[0]: Stage 1, fused[1]: Stage 2, fused[2]: Stage 3, fused[3]: Stage 4, fused[4]: Stage 5
        bottleneck = self.bottleneck(fused[4])

        # 3. U-Net decoder with skip connections
        d1 = self.decoder_block1(bottleneck, fused[3])
        d2 = self.decoder_block2(d1, fused[2])
        d3 = self.decoder_block3(d2, fused[1])
        d4 = self.decoder_block4(d3, fused[0])

        # Final upsample to match input spatial size H, W
        out_features = F.interpolate(d4, size=pre.shape[2:], mode="bilinear", align_corners=False)
        out_features = self.final_up(out_features)

        # 4. Task heads
        loc_logits = self.loc_head(out_features)
        dmg_logits = self.dmg_head(out_features)

        return loc_logits, dmg_logits
