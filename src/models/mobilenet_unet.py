import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """Efficient depthwise separable convolution block for low latency"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x


class InvertedResidualBlock(nn.Module):
    """MobileNetV3 style inverted residual block with expand ratio"""
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=3):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = []
        if expand_ratio != 1:
            # pw expansion
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish(inplace=True)
            ])
        # dw
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Hardswish(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetUNet(nn.Module):
    """
    Proposed Lightweight Segmentation Architecture (SegFormer-Lite / PIDNet / MobileNet Style).
    Designed for high FPS, low-latency edge deployment for Flood/Road Segmentation.
    """
    def __init__(self, in_channels=3, num_classes=3, feature_dim=128):
        super(MobileNetUNet, self).__init__()
        self.name = "MobileNet-UNet (Lite Proposed)"
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Stem block (Input -> 32)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(inplace=True)
        ) # (H/2, W/2)

        # Stage 1: (H/2 -> H/4)
        self.stage1 = nn.Sequential(
            InvertedResidualBlock(32, 32, stride=1, expand_ratio=2),
            InvertedResidualBlock(32, 48, stride=2, expand_ratio=3)
        ) # (H/4, W/4)

        # Stage 2: (H/4 -> H/8)
        self.stage2 = nn.Sequential(
            InvertedResidualBlock(48, 64, stride=2, expand_ratio=3),
            InvertedResidualBlock(64, 64, stride=1, expand_ratio=3)
        ) # (H/8, W/8)

        # Stage 3: (H/8 -> H/16)
        self.stage3 = nn.Sequential(
            InvertedResidualBlock(64, 96, stride=2, expand_ratio=4),
            InvertedResidualBlock(96, 128, stride=1, expand_ratio=4)
        ) # (H/16, W/16)

        # Feature Pyramid Fusion / Aggregator (SegFormer-Lite style MLP decoder)
        self.lin3 = nn.Conv2d(128, feature_dim, kernel_size=1)
        self.lin2 = nn.Conv2d(64, feature_dim, kernel_size=1)
        self.lin1 = nn.Conv2d(48, feature_dim, kernel_size=1)
        self.lin0 = nn.Conv2d(32, feature_dim, kernel_size=1)

        # Fusion head & Lightweight classifier
        self.fusion = DepthwiseSeparableConv(feature_dim * 4, feature_dim)
        self.classifier = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(feature_dim, num_classes, kernel_size=1)
        )

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        # Encoder forward pass
        s0 = self.stem(x)    # (H/2, W/2, 32)
        s1 = self.stage1(s0) # (H/4, W/4, 48)
        s2 = self.stage2(s1) # (H/8, W/8, 64)
        s3 = self.stage3(s2) # (H/16, W/16, 128)

        # Multi-scale feature alignment to H/4
        target_size = (s1.shape[2], s1.shape[3])

        p0 = F.interpolate(self.lin0(s0), size=target_size, mode='bilinear', align_corners=False)
        p1 = self.lin1(s1)
        p2 = F.interpolate(self.lin2(s2), size=target_size, mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.lin3(s3), size=target_size, mode='bilinear', align_corners=False)

        # Concatenate and fuse
        fused = torch.cat([p0, p1, p2, p3], dim=1)
        feat = self.fusion(fused)
        
        # Classifier & final interpolation to original input size
        logits = self.classifier(feat)
        out = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        return out
