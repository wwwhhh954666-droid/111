import argparse
import csv
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_black_grime_aug(img):
    out = img.astype(np.float32).copy()
    h, w = out.shape[:2]
    for _ in range(random.randint(1, 3)):
        mask = np.zeros((h, w), dtype=np.float32)
        center = (random.randint(0, max(0, w - 1)), random.randint(0, max(0, h - 1)))
        axes = (
            random.randint(max(8, w // 16), max(12, w // 3)),
            random.randint(max(8, h // 18), max(12, h // 3)),
        )
        angle = random.uniform(0, 180)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), random.uniform(7.0, 21.0))
        alpha = random.uniform(0.18, 0.46)
        color = np.array([random.randint(8, 45), random.randint(8, 45), random.randint(8, 45)], dtype=np.float32)
        out = out * (1.0 - alpha * mask[..., None]) + color * (alpha * mask[..., None])
    if random.random() < 0.45:
        out = np.clip(out * random.uniform(0.78, 0.95) + random.uniform(-10, 4), 0, 255)
    return out.astype(np.uint8)


class PatchDataset(Dataset):
    def __init__(self, root, split, augment=False, grime_aug_prob=0.0):
        self.root = Path(root) / split
        self.images = sorted((self.root / "images").glob("*.jpg"))
        self.augment = augment
        self.grime_aug_prob = grime_aug_prob

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.root / "masks" / f"{img_path.stem}.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if self.augment:
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[::-1])
                mask = np.ascontiguousarray(mask[::-1])
            if random.random() < 0.35:
                gain = random.uniform(0.8, 1.2)
                bias = random.uniform(-18, 18)
                img = np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
            if self.grime_aug_prob > 0 and random.random() < self.grime_aug_prob:
                img = apply_black_grime_aug(img)

        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        img = (img - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        mask = torch.from_numpy(mask.astype(np.int64))
        return img, mask


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, num_classes=3, base=32):
        super().__init__()
        self.e1 = ConvBlock(3, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 8)
        self.b = ConvBlock(base * 8, base * 8)
        self.d4 = ConvBlock(base * 16, base * 4)
        self.d3 = ConvBlock(base * 8, base * 2)
        self.d2 = ConvBlock(base * 4, base)
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        b = self.b(F.max_pool2d(e4, 2))
        x = F.interpolate(b, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        x = self.d4(torch.cat([x, e4], 1))
        x = F.interpolate(x, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.d3(torch.cat([x, e3], 1))
        x = F.interpolate(x, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.d2(torch.cat([x, e2], 1))
        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(x)


class TinyFPN(nn.Module):
    def __init__(self, num_classes=3, base=32):
        super().__init__()
        self.c1 = ConvBlock(3, base)
        self.c2 = ConvBlock(base, base * 2)
        self.c3 = ConvBlock(base * 2, base * 4)
        self.c4 = ConvBlock(base * 4, base * 8)
        self.l1 = nn.Conv2d(base, base, 1)
        self.l2 = nn.Conv2d(base * 2, base, 1)
        self.l3 = nn.Conv2d(base * 4, base, 1)
        self.l4 = nn.Conv2d(base * 8, base, 1)
        self.smooth = ConvBlock(base, base)
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(F.max_pool2d(c1, 2))
        c3 = self.c3(F.max_pool2d(c2, 2))
        c4 = self.c4(F.max_pool2d(c3, 2))
        p4 = self.l4(c4)
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p1 = self.l1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="nearest")
        return self.head(self.smooth(p1))


class ASPP(nn.Module):
    def __init__(self, ch, out_ch):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(ch, out_ch, 1),
            nn.Conv2d(ch, out_ch, 3, padding=2, dilation=2),
            nn.Conv2d(ch, out_ch, 3, padding=4, dilation=4),
            nn.Conv2d(ch, out_ch, 3, padding=8, dilation=8),
        ])
        self.out = ConvBlock(out_ch * 4, out_ch)

    def forward(self, x):
        return self.out(torch.cat([b(x) for b in self.branches], 1))


class TinyASPP(nn.Module):
    def __init__(self, num_classes=3, base=32):
        super().__init__()
        self.e1 = ConvBlock(3, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 8)
        self.aspp = ASPP(base * 8, base * 4)
        self.low = nn.Conv2d(base, base, 1)
        self.dec = ConvBlock(base * 5, base * 2)
        self.head = nn.Conv2d(base * 2, num_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        x = self.aspp(e4)
        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec(torch.cat([x, self.low(e1)], 1))
        return self.head(x)


class MixBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.attn(self.norm1(tokens), self.norm1(tokens), self.norm1(tokens), need_weights=False)[0]
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class SegFormerTiny(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.s1 = nn.Sequential(nn.Conv2d(3, 32, 7, stride=4, padding=3, bias=False), nn.BatchNorm2d(32), nn.ReLU(True))
        self.s2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True), MixBlock(64, 4))
        self.s3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True), MixBlock(128, 4))
        self.s4 = nn.Sequential(nn.Conv2d(128, 192, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(192), nn.ReLU(True), MixBlock(192, 6))
        self.proj1 = nn.Conv2d(32, 64, 1)
        self.proj2 = nn.Conv2d(64, 64, 1)
        self.proj3 = nn.Conv2d(128, 64, 1)
        self.proj4 = nn.Conv2d(192, 64, 1)
        self.fuse = ConvBlock(256, 96)
        self.head = nn.Conv2d(96, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[-2:]
        s1 = self.s1(x)
        s2 = self.s2(s1)
        s3 = self.s3(s2)
        s4 = self.s4(s3)
        size = s1.shape[-2:]
        feats = [
            self.proj1(s1),
            F.interpolate(self.proj2(s2), size=size, mode="bilinear", align_corners=False),
            F.interpolate(self.proj3(s3), size=size, mode="bilinear", align_corners=False),
            F.interpolate(self.proj4(s4), size=size, mode="bilinear", align_corners=False),
        ]
        out = self.head(self.fuse(torch.cat(feats, 1)))
        return F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)


class HRNetLite(nn.Module):
    def __init__(self, num_classes=3, base=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(True),
            nn.Conv2d(base, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(True),
        )
        self.high1 = ConvBlock(base, base)
        self.low1 = ConvBlock(base, base * 2)
        self.high2 = ConvBlock(base + base * 2, base)
        self.low2 = ConvBlock(base * 2 + base, base * 2)
        self.high3 = ConvBlock(base + base * 2, base)
        self.low3 = ConvBlock(base * 2 + base, base * 2)
        self.head = nn.Sequential(ConvBlock(base + base * 2, base * 2), nn.Conv2d(base * 2, num_classes, 1))

    def forward(self, x):
        h0 = self.stem(x)
        h = self.high1(h0)
        l = self.low1(F.max_pool2d(h0, 2))
        h = self.high2(torch.cat([h, F.interpolate(l, size=h.shape[-2:], mode="bilinear", align_corners=False)], 1))
        l = self.low2(torch.cat([l, F.max_pool2d(h, 2)], 1))
        h = self.high3(torch.cat([h, F.interpolate(l, size=h.shape[-2:], mode="bilinear", align_corners=False)], 1))
        l = self.low3(torch.cat([l, F.max_pool2d(h, 2)], 1))
        return self.head(torch.cat([h, F.interpolate(l, size=h.shape[-2:], mode="bilinear", align_corners=False)], 1))


class OCRLite(nn.Module):
    def __init__(self, num_classes=3, base=32):
        super().__init__()
        self.e1 = ConvBlock(3, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.aux = nn.Conv2d(base * 4, num_classes, 1)
        self.query = nn.Conv2d(base * 4, base * 4, 1)
        self.out = ConvBlock(base * 8, base * 2)
        self.head = nn.Conv2d(base * 2, num_classes, 1)

    def forward(self, x):
        size = x.shape[-2:]
        f1 = self.e1(x)
        f2 = self.e2(F.max_pool2d(f1, 2))
        feat = self.e3(F.max_pool2d(f2, 2))
        aux = self.aux(feat)
        prob = torch.softmax(aux, dim=1)
        b, c, h, w = feat.shape
        feat_flat = feat.reshape(b, c, -1)
        prob_flat = prob.reshape(b, prob.shape[1], -1)
        context = torch.bmm(feat_flat, prob_flat.transpose(1, 2)) / (prob_flat.sum(-1).unsqueeze(1) + 1e-6)
        sim = torch.bmm(self.query(feat).reshape(b, c, -1).transpose(1, 2), context)
        sim = torch.softmax(sim / math.sqrt(c), dim=-1)
        obj_context = torch.bmm(context, sim.transpose(1, 2)).reshape(b, c, h, w)
        out = self.head(self.out(torch.cat([feat, obj_context], 1)))
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False)


class TorchvisionSegWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


class BinaryLRASPPHead(nn.Module):
    def __init__(self, low_channels=40, high_channels=960, out_channels=1):
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(high_channels, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, 128, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(low_channels, out_channels, 1)
        self.high_classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, features):
        low = features["low"]
        high = features["high"]
        x = self.cbr(high)
        x = x * self.scale(high)
        x = self.high_classifier(x)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        return x + self.low_classifier(low)


class DualLRASPP(nn.Module):
    def __init__(self):
        super().__init__()
        base = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
            weights=None, weights_backbone=None, num_classes=3
        )
        self.backbone = base.backbone
        self.crack_head = BinaryLRASPPHead()
        self.spall_head = BinaryLRASPPHead()

    def forward(self, x):
        size = x.shape[-2:]
        features = self.backbone(x)
        crack_logit = self.crack_head(features)
        spall_logit = self.spall_head(features)
        crack_logit = F.interpolate(crack_logit, size=size, mode="bilinear", align_corners=False)
        spall_logit = F.interpolate(spall_logit, size=size, mode="bilinear", align_corners=False)
        background_logit = torch.zeros_like(crack_logit)
        return torch.cat([background_logit, crack_logit, spall_logit], dim=1)


class DualLRASPPAux(nn.Module):
    def __init__(self):
        super().__init__()
        base = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
            weights=None, weights_backbone=None, num_classes=3
        )
        self.backbone = base.backbone
        self.crack_head = BinaryLRASPPHead()
        self.spall_head = BinaryLRASPPHead()
        self.bg_head = BinaryLRASPPHead()

    def forward(self, x):
        size = x.shape[-2:]
        features = self.backbone(x)
        crack = F.interpolate(self.crack_head(features), size=size, mode="bilinear", align_corners=False)
        spall = F.interpolate(self.spall_head(features), size=size, mode="bilinear", align_corners=False)
        bg = F.interpolate(self.bg_head(features), size=size, mode="bilinear", align_corners=False)
        out = torch.cat([bg, crack, spall], dim=1)
        return {"out": out, "crack": crack, "spall": spall}


class FixedEdgeCues(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        rgb = (x * std + mean).clamp(0.0, 1.0)
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        j11 = F.avg_pool2d(gx * gx, 9, stride=1, padding=4)
        j22 = F.avg_pool2d(gy * gy, 9, stride=1, padding=4)
        j12 = F.avg_pool2d(gx * gy, 9, stride=1, padding=4)
        coherence = torch.sqrt((j11 - j22).pow(2) + 4.0 * j12.pow(2) + 1e-6) / (j11 + j22 + 1e-6)
        return torch.cat([mag.clamp(0.0, 1.0), coherence.clamp(0.0, 1.0)], dim=1)


class FixedFailureCues(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge = FixedEdgeCues()

    def forward(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        rgb = (x * std + mean).clamp(0.0, 1.0)
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        mag, coherence = self.edge(x).chunk(2, dim=1)

        local = F.avg_pool2d(gray, 31, stride=1, padding=15)
        local2 = F.avg_pool2d(gray * gray, 31, stride=1, padding=15)
        local_std = torch.sqrt((local2 - local * local).clamp_min(0.0) + 1e-6)
        dark_line = ((local - gray).clamp_min(0.0) / (local_std + 0.08)).clamp(0.0, 1.0)
        grime = torch.sigmoid((0.36 - gray) * 12.0)
        edge_density = F.avg_pool2d(mag, 17, stride=1, padding=8).clamp(0.0, 1.0)
        clutter = (edge_density * (1.0 - coherence)).clamp(0.0, 1.0)
        return torch.cat([mag, coherence, dark_line, grime, edge_density, clutter], dim=1)


class FailureCueFusion(nn.Module):
    def __init__(self, low_channels=40, cue_channels=6):
        super().__init__()
        self.cues = FixedFailureCues()
        self.shared = nn.Sequential(
            nn.Conv2d(low_channels + cue_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.suppress_head = nn.Conv2d(24, 1, 1)
        self.dark_head = nn.Conv2d(24, 1, 1)
        self.spall_boundary_head = nn.Conv2d(24, 1, 1)
        self.suppress_strength = nn.Parameter(torch.tensor(-2.2))
        self.dark_strength = nn.Parameter(torch.tensor(1.2))

    def forward(self, image, low_feature, out_size):
        cues = F.interpolate(self.cues(image), size=low_feature.shape[-2:], mode="bilinear", align_corners=False)
        feat = self.shared(torch.cat([low_feature, cues], dim=1))
        suppress = torch.sigmoid(self.suppress_head(feat))
        dark_line = torch.sigmoid(self.dark_head(feat))
        spall_boundary = self.spall_boundary_head(feat)
        return {
            "regular": F.interpolate(suppress, size=out_size, mode="bilinear", align_corners=False),
            "dark_line": F.interpolate(dark_line, size=out_size, mode="bilinear", align_corners=False),
            "spall_boundary": F.interpolate(spall_boundary, size=out_size, mode="bilinear", align_corners=False),
        }


class RegularStructureGate(nn.Module):
    def __init__(self, low_channels=40):
        super().__init__()
        self.edge = FixedEdgeCues()
        self.gate = nn.Sequential(
            nn.Conv2d(low_channels + 2, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )
        self.strength = nn.Parameter(torch.tensor(1.0))

    def forward(self, image, low_feature, out_size):
        cues = F.interpolate(self.edge(image), size=low_feature.shape[-2:], mode="bilinear", align_corners=False)
        gate = torch.sigmoid(self.gate(torch.cat([low_feature, cues], dim=1)))
        gate = F.interpolate(gate, size=out_size, mode="bilinear", align_corners=False)
        return gate


class DualLRASPPAuxRSG(nn.Module):
    def __init__(self):
        super().__init__()
        base = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
            weights=None, weights_backbone=None, num_classes=3
        )
        self.backbone = base.backbone
        self.crack_head = BinaryLRASPPHead()
        self.spall_head = BinaryLRASPPHead()
        self.bg_head = BinaryLRASPPHead()
        self.regular_gate = RegularStructureGate()

    def forward(self, x):
        size = x.shape[-2:]
        features = self.backbone(x)
        raw_crack = F.interpolate(self.crack_head(features), size=size, mode="bilinear", align_corners=False)
        spall = F.interpolate(self.spall_head(features), size=size, mode="bilinear", align_corners=False)
        bg = F.interpolate(self.bg_head(features), size=size, mode="bilinear", align_corners=False)
        regular = self.regular_gate(x, features["low"], size)
        crack = raw_crack - F.softplus(self.regular_gate.strength) * regular
        out = torch.cat([bg, crack, spall], dim=1)
        return {"out": out, "crack": crack, "spall": spall, "raw_crack": raw_crack, "regular": regular}


class DualLRASPPAuxFCF(nn.Module):
    def __init__(self):
        super().__init__()
        base = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
            weights=None, weights_backbone=None, num_classes=3
        )
        self.backbone = base.backbone
        self.crack_head = BinaryLRASPPHead()
        self.spall_head = BinaryLRASPPHead()
        self.bg_head = BinaryLRASPPHead()
        self.regular_gate = RegularStructureGate()
        self.failure_fusion = FailureCueFusion()

    def forward(self, x):
        size = x.shape[-2:]
        features = self.backbone(x)
        raw_crack = F.interpolate(self.crack_head(features), size=size, mode="bilinear", align_corners=False)
        spall = F.interpolate(self.spall_head(features), size=size, mode="bilinear", align_corners=False)
        bg = F.interpolate(self.bg_head(features), size=size, mode="bilinear", align_corners=False)
        regular = self.regular_gate(x, features["low"], size)
        cues = self.failure_fusion(x, features["low"], size)

        weak_crack = 1.0 - torch.sigmoid(raw_crack)
        suppress = F.softplus(self.regular_gate.strength) * regular * weak_crack
        dark_boost = F.softplus(self.failure_fusion.dark_strength) * cues["dark_line"]
        crack = raw_crack + dark_boost - suppress
        out = torch.cat([bg, crack, spall], dim=1)
        return {
            "out": out,
            "crack": crack,
            "spall": spall,
            "raw_crack": raw_crack,
            "regular": regular,
            "dark_line": cues["dark_line"],
            "spall_boundary": cues["spall_boundary"],
        }


def make_model(name):
    if name == "unet":
        return TinyUNet()
    if name == "fpn":
        return TinyFPN()
    if name == "aspp":
        return TinyASPP()
    if name == "tv_deeplab_mnv3":
        model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
            weights=None, weights_backbone=None, num_classes=3
        )
        return TorchvisionSegWrapper(model)
    if name == "tv_lraspp_mnv3":
        model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
            weights=None, weights_backbone=None, num_classes=3
        )
        return TorchvisionSegWrapper(model)
    if name == "dual_lraspp_mnv3":
        return DualLRASPP()
    if name == "dual_lraspp_aux_mnv3":
        return DualLRASPPAux()
    if name == "dual_lraspp_aux_rsg_mnv3":
        return DualLRASPPAuxRSG()
    if name == "dual_lraspp_aux_fcf_mnv3":
        return DualLRASPPAuxFCF()
    if name == "segformer_tiny":
        return SegFormerTiny()
    if name == "hrnet_lite":
        return HRNetLite()
    if name == "ocr_lite":
        return OCRLite()
    raise ValueError(name)


def dice_loss(logits, target, num_classes=3):
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target.clamp(0, num_classes - 1), num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = torch.sum(probs * one_hot, dims)
    union = torch.sum(probs + one_hot, dims)
    dice = (2 * inter + 1.0) / (union + 1.0)
    return 1.0 - dice[1:].mean()


def focal_loss(logits, target, alpha=None, gamma=2.0):
    ce = F.cross_entropy(logits, target, weight=alpha, reduction="none")
    pt = torch.exp(-ce)
    return (((1.0 - pt) ** gamma) * ce).mean()


def spall_boundary_target(mask, radius=2):
    binary = (mask == 2).float().unsqueeze(1)
    kernel = radius * 2 + 1
    dilated = F.max_pool2d(binary, kernel, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - binary, kernel, stride=1, padding=radius)
    return (dilated - eroded).clamp(0.0, 1.0)


def soft_boundary_cross_entropy(logits, target, ce_weight=None, boundary_weight=0.45):
    ce = F.cross_entropy(logits, target, weight=ce_weight, reduction="none")
    boundary = spall_boundary_target(target).squeeze(1)
    pixel_weight = torch.ones_like(ce)
    pixel_weight = torch.where(boundary > 0, pixel_weight * boundary_weight, pixel_weight)
    return (ce * pixel_weight).sum() / pixel_weight.sum().clamp_min(1.0)


def soft_boundary_focal_loss(logits, target, alpha=None, gamma=2.0, boundary_weight=0.45):
    ce = F.cross_entropy(logits, target, weight=alpha, reduction="none")
    pt = torch.exp(-ce)
    focal = ((1.0 - pt) ** gamma) * ce
    boundary = spall_boundary_target(target).squeeze(1)
    pixel_weight = torch.ones_like(focal)
    pixel_weight = torch.where(boundary > 0, pixel_weight * boundary_weight, pixel_weight)
    return (focal * pixel_weight).sum() / pixel_weight.sum().clamp_min(1.0)


def balanced_bce_with_logits(logit, target):
    pos = target.sum()
    neg = target.numel() - pos
    pos_weight = (neg / (pos + 1.0)).clamp(1.0, 40.0)
    return F.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_weight)


def tversky_loss(logits, target, num_classes=3, alpha=0.7, beta=0.3):
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target.clamp(0, num_classes - 1), num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    tp = torch.sum(probs * one_hot, dims)
    fp = torch.sum(probs * (1.0 - one_hot), dims)
    fn = torch.sum((1.0 - probs) * one_hot, dims)
    score = (tp + 1.0) / (tp + alpha * fp + beta * fn + 1.0)
    return 1.0 - score[1:].mean()


def compute_loss(
    logits,
    mask,
    mode,
    ce_weight=None,
    num_classes=3,
    soft_boundary_weight=0.45,
    boundary_aux_weight=0.15,
    gate_l1_weight=0.01,
):
    if isinstance(logits, dict):
        out = logits["out"]
        final_loss = compute_loss(
            out,
            mask,
            mode,
            ce_weight,
            num_classes,
            soft_boundary_weight,
            boundary_aux_weight,
            gate_l1_weight,
        )
        crack_target = (mask == 1).float().unsqueeze(1)
        spall_target = (mask == 2).float().unsqueeze(1)
        crack_loss = binary_focal_tversky_loss(logits["crack"], crack_target)
        spall_loss = binary_focal_tversky_loss(logits["spall"], spall_target)
        loss = final_loss + 0.5 * crack_loss + 0.5 * spall_loss
        if "spall_boundary" in logits and boundary_aux_weight > 0:
            loss = loss + boundary_aux_weight * balanced_bce_with_logits(logits["spall_boundary"], spall_boundary_target(mask))
        if gate_l1_weight > 0:
            if "regular" in logits:
                loss = loss + gate_l1_weight * logits["regular"].mean()
            if "dark_line" in logits:
                loss = loss + gate_l1_weight * logits["dark_line"].mean()
        return loss
    if mode == "ce_dice":
        return F.cross_entropy(logits, mask, weight=ce_weight) + dice_loss(logits, mask, num_classes)
    if mode == "focal_dice":
        return focal_loss(logits, mask, alpha=ce_weight) + dice_loss(logits, mask, num_classes)
    if mode == "ce_tversky":
        return F.cross_entropy(logits, mask, weight=ce_weight) + tversky_loss(logits, mask, num_classes)
    if mode == "focal_tversky":
        return focal_loss(logits, mask, alpha=ce_weight) + tversky_loss(logits, mask, num_classes)
    if mode == "soft_ce_tversky":
        return soft_boundary_cross_entropy(logits, mask, ce_weight, soft_boundary_weight) + tversky_loss(logits, mask, num_classes)
    if mode == "soft_focal_tversky":
        return soft_boundary_focal_loss(logits, mask, ce_weight, boundary_weight=soft_boundary_weight) + tversky_loss(logits, mask, num_classes)
    raise ValueError(f"Unknown loss mode: {mode}")


def binary_focal_tversky_loss(logit, target, alpha=0.7, beta=0.3, gamma=1.0):
    prob = torch.sigmoid(logit)
    dims = (0, 2, 3)
    tp = torch.sum(prob * target, dims)
    fp = torch.sum(prob * (1.0 - target), dims)
    fn = torch.sum((1.0 - prob) * target, dims)
    tversky = (tp + 1.0) / (tp + alpha * fp + beta * fn + 1.0)
    return torch.pow(1.0 - tversky, gamma).mean()


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    num_classes=3,
    loss_mode="ce_dice",
    ce_weight=None,
    soft_boundary_weight=0.45,
    boundary_aux_weight=0.15,
    gate_l1_weight=0.01,
):
    model.eval()
    conf = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)
    val_loss = 0.0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        logits = model(img)
        loss = compute_loss(
            logits,
            mask,
            loss_mode,
            ce_weight,
            num_classes,
            soft_boundary_weight,
            boundary_aux_weight,
            gate_l1_weight,
        )
        val_loss += loss.item()
        pred_logits = logits["out"] if isinstance(logits, dict) else logits
        pred = pred_logits.argmax(1)
        idx = mask * num_classes + pred
        conf += torch.bincount(idx.flatten(), minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    ious = []
    dices = []
    for c in range(num_classes):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        iou = tp / (tp + fp + fn + 1e-9)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-9)
        ious.append(float(iou.cpu()))
        dices.append(float(dice.cpu()))
    return {
        "val_loss": val_loss / max(1, len(loader)),
        "miou_all": float(np.mean(ious)),
        "miou_damage": float(np.mean(ious[1:])),
        "iou_crack": ious[1],
        "iou_spall": ious[2],
        "dice_crack": dices[1],
        "dice_spall": dices[2],
    }


def save_preview(model, dataset, out_dir, device, max_items=6):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    colors = np.array([[0, 0, 0], [255, 40, 40], [40, 180, 255]], dtype=np.uint8)
    for i in range(min(max_items, len(dataset))):
        img, mask = dataset[i]
        with torch.no_grad():
            logits = model(img.unsqueeze(0).to(device))
            logits = logits["out"] if isinstance(logits, dict) else logits
            pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        denorm = img * torch.tensor([0.229, 0.224, 0.225])[:, None, None] + torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        rgb = np.clip(denorm.numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        gt_overlay = (rgb * 0.65 + colors[mask.numpy()] * 0.35).astype(np.uint8)
        pred_overlay = (rgb * 0.65 + colors[pred] * 0.35).astype(np.uint8)
        panel = np.concatenate([rgb, gt_overlay, pred_overlay], axis=1)
        cv2.imwrite(str(out_dir / f"preview_{i:02d}.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])


def class_weights_from_arg(arg, device):
    if not arg:
        return None
    values = [float(x) for x in arg.split(",")]
    if len(values) != 3:
        raise ValueError("--class-weights must contain 3 comma-separated values")
    return torch.tensor(values, dtype=torch.float32, device=device)


def train_one(model_name, args, device):
    out_dir = Path(args.out) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds = PatchDataset(args.data, "train", augment=True, grime_aug_prob=args.grime_aug_prob)
    val_ds = PatchDataset(args.data, "val", augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = make_model(model_name).to(device)
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"initialized with partial checkpoint: missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )
    if args.train_failure_fusion_only:
        for param in model.parameters():
            param.requires_grad = False
        if not hasattr(model, "failure_fusion"):
            raise ValueError("--train-failure-fusion-only requires a model with failure_fusion")
        for param in model.failure_fusion.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"trainable_params={trainable}", flush=True)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ce_weight = class_weights_from_arg(args.class_weights, device)
    best = {"miou_damage": -1.0}
    history = []

    if args.eval_initial:
        metrics = evaluate(
            model,
            val_loader,
            device,
            loss_mode=args.loss_mode,
            ce_weight=ce_weight,
            soft_boundary_weight=args.soft_boundary_weight,
            boundary_aux_weight=args.boundary_aux_weight,
            gate_l1_weight=args.gate_l1_weight,
        )
        metrics.update({"epoch": 0, "train_loss": 0.0})
        history.append(metrics)
        print(model_name, json.dumps(metrics, ensure_ascii=False))
        best = dict(metrics)
        torch.save({"model": model.state_dict(), "metrics": best, "name": model_name}, out_dir / "best.pt")
        save_preview(model, val_ds, out_dir / "previews", device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(img)
                loss = compute_loss(
                    logits,
                    mask,
                    args.loss_mode,
                    ce_weight,
                    soft_boundary_weight=args.soft_boundary_weight,
                    boundary_aux_weight=args.boundary_aux_weight,
                    gate_l1_weight=args.gate_l1_weight,
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
        metrics = evaluate(
            model,
            val_loader,
            device,
            loss_mode=args.loss_mode,
            ce_weight=ce_weight,
            soft_boundary_weight=args.soft_boundary_weight,
            boundary_aux_weight=args.boundary_aux_weight,
            gate_l1_weight=args.gate_l1_weight,
        )
        metrics.update({"epoch": epoch, "train_loss": running / max(1, len(train_loader))})
        history.append(metrics)
        print(model_name, json.dumps(metrics, ensure_ascii=False))
        if metrics["miou_damage"] > best["miou_damage"]:
            best = dict(metrics)
            torch.save({"model": model.state_dict(), "metrics": best, "name": model_name}, out_dir / "best.pt")
            save_preview(model, val_ds, out_dir / "previews", device)

    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    (out_dir / "best_metrics.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=r"D:\mask\baseline_runs\patch_dataset")
    parser.add_argument("--out", default=r"D:\mask\baseline_runs\experiments")
    parser.add_argument("--models", nargs="+", default=["unet", "fpn", "aspp"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--class-weights", default="")
    parser.add_argument(
        "--loss-mode",
        default="ce_dice",
        choices=["ce_dice", "focal_dice", "ce_tversky", "focal_tversky", "soft_ce_tversky", "soft_focal_tversky"],
    )
    parser.add_argument("--grime-aug-prob", type=float, default=0.0)
    parser.add_argument("--soft-boundary-weight", type=float, default=0.45)
    parser.add_argument("--boundary-aux-weight", type=float, default=0.15)
    parser.add_argument("--gate-l1-weight", type=float, default=0.01)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--eval-initial", action="store_true")
    parser.add_argument("--train-failure-fusion-only", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    results = []
    for model_name in args.models:
        best = train_one(model_name, args, device)
        best["model"] = model_name
        results.append(best)
    results = sorted(results, key=lambda x: x["miou_damage"], reverse=True)
    with (Path(args.out) / "baseline_summary.csv").open("w", newline="", encoding="utf-8") as f:
        keys = ["model", "epoch", "miou_damage", "iou_crack", "iou_spall", "dice_crack", "dice_spall", "val_loss"]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in keys})
    print("SUMMARY")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
