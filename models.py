import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import time


class AMSoftmax(nn.Module):
    def __init__(self, in_feats, n_classes=2, m=0.15, s=20.0):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.W = nn.Parameter(torch.randn(in_feats, n_classes))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, label=None):
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.W, p=2, dim=0)
        costh = torch.mm(x_norm, w_norm)

        costh = torch.clamp(costh, -1.0 + 1e-7, 1.0 - 1e-7)

        if label is not None:
            if type(label) is tuple:
                label_a, label_b = label
                delt_costh_a = torch.zeros_like(costh).scatter_(1, label_a.view(-1, 1), self.m)
                delt_costh_b = torch.zeros_like(costh).scatter_(1, label_b.view(-1, 1), self.m)
                costh_m_s_a = self.s * (costh - delt_costh_a)
                costh_m_s_b = self.s * (costh - delt_costh_b)
                return costh_m_s_a, costh_m_s_b
            else:
                delt_costh = torch.zeros_like(costh).scatter_(1, label.view(-1, 1), self.m)
                return self.s * (costh - delt_costh)

        return self.s * costh


class StatsPool(nn.Module):
    def __init__(self):
        super(StatsPool, self).__init__()
        self.proj = nn.Linear(1024, 512)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        mean = x.mean(dim=2)
        var = torch.var(x, dim=2, unbiased=False)
        std = torch.sqrt(torch.clamp(var, min=1e-8) + 1e-6)
        pooled = torch.cat([mean, std], dim=1)
        return self.proj(pooled)


# ==========================================
# THE FEATURE EXTRACTOR (BACKBONE)
# ==========================================
class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim=512, bottleneck_dim=128):
        super(AttentiveStatsPool, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck_dim),
            nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1),
            nn.Softmax(dim=2)
        )

        self.proj = nn.Linear(in_dim * 2, 512)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        alpha = self.attention(x)
        mean = torch.sum(alpha * x, dim=2)
        residuals = x - mean.unsqueeze(2)
        var = torch.sum(alpha * (residuals ** 2), dim=2)
        std = torch.sqrt(torch.clamp(var, min=1e-8))
        pooled = torch.cat([mean, std], dim=1)
        return self.proj(pooled)


class DualStatsPool(nn.Module):
    """Combines Global StatsPool + Attentive StatsPool for big-picture AND localized features."""
    def __init__(self, in_dim=512, bottleneck_dim=128):
        super(DualStatsPool, self).__init__()
        self.global_pool = StatsPool()
        self.attn_pool = AttentiveStatsPool(in_dim, bottleneck_dim)
        self.proj = nn.Linear(1024, 512)

    def forward(self, x):
        x_flat = x.view(x.size(0), x.size(1), -1)
        g = self.global_pool(x)
        a = self.attn_pool(x_flat)
        return self.proj(torch.cat([g, a], dim=1))


class BoundaryAnomalyPool(nn.Module):
    """
    Multi-scale boundary and anomaly pooling designed to detect localized unit-selection
    concatenation artifacts (A12) alongside global deepfake distortions.
    Combines:
      1. Global Mean: captures global spectral envelope distribution
      2. Global Variance/Std: captures overall spectral energy spread
      3. Temporal Peak (Max): detects worst-case anomalous frames anywhere in the utterance
      4. Temporal Delta Max: detects abrupt frame-to-frame boundary discontinuities (splice points)
    """
    def __init__(self, in_dim=512):
        super(BoundaryAnomalyPool, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512)
        )

    def forward(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), x.size(1), -1)  # (B, C, T)

        # 1. Global Mean
        mean = x.mean(dim=2)

        # 2. Global Std
        var = torch.var(x, dim=2, unbiased=False)
        std = torch.sqrt(torch.clamp(var, min=1e-8) + 1e-6)

        # 3. Peak Local Anomaly
        max_val, _ = x.max(dim=2)

        # 4. Local Boundary Discontinuity (frame-to-frame delta jump)
        if x.size(2) > 1:
            delta = torch.diff(x, dim=2).abs()
            delta_max, _ = delta.max(dim=2)
        else:
            delta_max = torch.zeros_like(max_val)

        pooled = torch.cat([mean, std, max_val, delta_max], dim=1)
        return self.proj(pooled)


class AudioResNet(nn.Module):
    def __init__(self, use_asp=False, pool_type=None):
        super(AudioResNet, self).__init__()
        self.backbone = resnet18(weights=None)

        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=1, padding=3, bias=False
        )

        with torch.no_grad():
            self.backbone.conv1.weight = nn.Parameter(
                original_conv1.weight.sum(dim=1, keepdim=True)
            )

        self.backbone.maxpool = nn.Identity()

        if pool_type == 'boundary':
            self.backbone.avgpool = BoundaryAnomalyPool(in_dim=512)
        elif pool_type == 'dual' or use_asp:
            self.backbone.avgpool = DualStatsPool(in_dim=512)
        else:
            self.backbone.avgpool = StatsPool()

        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.dropout(x)
        x = self.backbone.layer3(x)
        x = self.dropout(x)
        x = self.backbone.layer4(x)
        x = self.dropout(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        return x


# ==========================================
# BASELINE MODELS
# ==========================================
class LFCCModel(nn.Module):
    def __init__(self, use_asp=False, pool_type=None):
        super(LFCCModel, self).__init__()
        self.lfcc_net = AudioResNet(use_asp=use_asp, pool_type=pool_type)
        self.dropout = nn.Dropout(0.5)
        self.classifier = AMSoftmax(512, 2)

    def forward(self, mfcc, lfcc, label=None):
        lfcc = lfcc.unsqueeze(1) if lfcc.dim() == 3 else lfcc
        emb = self.lfcc_net(lfcc)
        return self.classifier(self.dropout(emb), label)


class MFCCModel(nn.Module):
    def __init__(self, use_asp=False, pool_type=None):
        super(MFCCModel, self).__init__()
        self.mfcc_net = AudioResNet(use_asp=use_asp, pool_type=pool_type)
        self.dropout = nn.Dropout(0.5)
        self.classifier = AMSoftmax(512, 2)

    def forward(self, mfcc, lfcc, label=None):
        mfcc = mfcc.unsqueeze(1) if mfcc.dim() == 3 else mfcc
        emb = self.mfcc_net(mfcc)
        return self.classifier(self.dropout(emb), label)


# ==========================================
# STATIC FUSION (CONCAT)
# ==========================================
class ConcatFusionModel(nn.Module):
    def __init__(self, use_asp=False, pool_type=None):
        super(ConcatFusionModel, self).__init__()
        self.lfcc_net = AudioResNet(use_asp=use_asp, pool_type=pool_type)
        self.mfcc_net = AudioResNet(use_asp=use_asp, pool_type=pool_type)
        self.dropout = nn.Dropout(0.5)
        self.classifier = AMSoftmax(1024, 2)

    def forward(self, mfcc, lfcc, label=None):
        lfcc = lfcc.unsqueeze(1) if lfcc.dim() == 3 else lfcc
        mfcc = mfcc.unsqueeze(1) if mfcc.dim() == 3 else mfcc
        emb_lfcc = self.lfcc_net(lfcc)
        emb_mfcc = self.mfcc_net(mfcc)
        emb = torch.cat((emb_mfcc, emb_lfcc), dim=1)
        return self.classifier(self.dropout(emb), label)


# ==========================================
# MLP FUSION (Difference + Bilinear)
# ==========================================
class MLPFusionModel(nn.Module):
    def __init__(self, use_asp=False, pool_type=None):
        super(MLPFusionModel, self).__init__()
        self.lfcc_net = AudioResNet(use_asp=use_asp, pool_type=pool_type)
        self.mfcc_net = AudioResNet(use_asp=use_asp, pool_type=pool_type)

        # Feature transformation
        self.mfcc_transform = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.lfcc_transform = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # Final MLP - Now takes 512 * 4 = 2048 dims
        self.mlp = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.6)
        )

        self.classifier = AMSoftmax(256, 2)

    def forward(self, mfcc, lfcc, label=None):
        lfcc = lfcc.unsqueeze(1) if lfcc.dim() == 3 else lfcc
        mfcc = mfcc.unsqueeze(1) if mfcc.dim() == 3 else mfcc

        emb_lfcc = self.lfcc_net(lfcc)
        emb_mfcc = self.mfcc_net(mfcc)

        # Enhance independent features
        t_mfcc = self.mfcc_transform(emb_mfcc)
        t_lfcc = self.lfcc_transform(emb_lfcc)

        # 1. Absolute Difference
        diff = torch.abs(t_mfcc - t_lfcc)

        # 2. Cross Multiplication
        interaction = t_mfcc * t_lfcc

        # Concat original transformed, difference, and interaction
        fused_all = torch.cat(
            (t_mfcc, t_lfcc, diff, interaction),
            dim=1
        )

        features = self.mlp(fused_all)
        return self.classifier(features, label)


# ==========================================
# ATTENTION FUSION
# ==========================================
class AttentionFusionModel(nn.Module):
    def __init__(self, use_asp=False, pool_type=None):
        super(AttentionFusionModel, self).__init__()
        self.lfcc_net = AudioResNet(use_asp=use_asp, pool_type=pool_type)
        self.mfcc_net = AudioResNet(use_asp=use_asp, pool_type=pool_type)
        self.modality_emb = nn.Parameter(torch.randn(1, 2, 512))
        self.attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.5)
        self.classifier = AMSoftmax(1024, 2)

    def forward(self, mfcc, lfcc, label=None):
        lfcc = lfcc.unsqueeze(1) if lfcc.dim() == 3 else lfcc
        mfcc = mfcc.unsqueeze(1) if mfcc.dim() == 3 else mfcc

        emb_lfcc = self.lfcc_net(lfcc)
        emb_mfcc = self.mfcc_net(mfcc)

        # Shape: (Batch, Seq_len, Embed_dim) -> (Batch, 2, 512)
        stacked_emb = torch.stack((emb_mfcc, emb_lfcc), dim=1)

        # Add modality embeddings
        stacked_emb = stacked_emb + self.modality_emb

        attn_out, _ = self.attention(
            stacked_emb,
            stacked_emb,
            stacked_emb
        )

        attn_out = attn_out.reshape(
            attn_out.size(0),
            -1
        )

        return self.classifier(
            self.dropout(attn_out),
            label
        )


# ==========================================
# STAGE 1 WAVEFORM BRANCH
# ==========================================
class WaveformBranch(nn.Module):
    """
    Learns local time-domain information directly from raw audio.

    Input:
        (B, N)
        or
        (B, 1, N)

    Output:
        (B, 256)
    """

    def __init__(self, out_dim=256):
        super(WaveformBranch, self).__init__()

        self.frontend = nn.Sequential(
            nn.Conv1d(
                1,
                64,
                kernel_size=80,
                stride=4,
                padding=38,
                bias=False
            ),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Conv1d(
                64,
                128,
                kernel_size=25,
                stride=4,
                padding=12,
                bias=False
            ),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(
                128,
                256,
                kernel_size=9,
                stride=2,
                padding=4,
                bias=False
            ),
            nn.BatchNorm1d(256),
            nn.GELU(),

            nn.Conv1d(
                256,
                256,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False
            ),
            nn.BatchNorm1d(256),
            nn.GELU()
        )

        # Multi-scale pooling: Local attention mean/std + temporal max anomaly + boundary delta jump
        self.local_attention = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(128, 1, kernel_size=1)
        )

        # 256 * 4 = 1024 dims (mean, std, max, delta_max) -> out_dim
        self.proj = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )

    def forward(self, waveform):
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        if waveform.dim() != 3:
            raise ValueError(
                f"Expected waveform shape (B, N) or (B, 1, N), "
                f"got {tuple(waveform.shape)}"
            )

        x = self.frontend(waveform)

        # x: (B, 256, T)
        # 1. Attentive local mean and std
        attention_logits = self.local_attention(x)
        attention = torch.softmax(attention_logits, dim=2)

        mean = torch.sum(attention * x, dim=2)
        residuals = x - mean.unsqueeze(2)
        var = torch.sum(attention * (residuals ** 2), dim=2)
        std = torch.sqrt(torch.clamp(var, min=1e-8))

        # 2. Peak local anomaly (Max across time)
        max_val, _ = x.max(dim=2)

        # 3. Local boundary jump (Delta max across time)
        if x.size(2) > 1:
            delta = torch.diff(x, dim=2).abs()
            delta_max, _ = delta.max(dim=2)
        else:
            delta_max = torch.zeros_like(max_val)

        # Keep mean, variability, peak anomaly, and boundary discontinuities
        pooled = torch.cat([mean, std, max_val, delta_max], dim=1)
        x = self.proj(pooled)
        return x


# ==========================================
# WAVEFORM FUSION MODEL
# ==========================================
class WaveformFusionModel(nn.Module):
    def __init__(self, use_asp=False, pool_type=None, freeze_backbones=False):
        super(WaveformFusionModel, self).__init__()
        self.freeze_backbones = freeze_backbones

        # Spectral branches with configurable pooling (e.g. boundary-aware)
        self.lfcc_net = AudioResNet(use_asp=use_asp, pool_type=pool_type)
        self.mfcc_net = AudioResNet(use_asp=use_asp, pool_type=pool_type)

        # Boundary-sensitive raw waveform branch
        self.wave_net = WaveformBranch(out_dim=256)

        # Transform spectral embeddings
        self.mfcc_transform = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU()
        )

        self.lfcc_transform = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU()
        )

        # 256 MFCC + 256 LFCC + 256 Difference + 256 Interaction + 256 Waveform = 1280
        self.fusion = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        self.classifier = AMSoftmax(256, 2)

    def forward(self, mfcc, lfcc, waveform, label=None):
        if mfcc.dim() == 3:
            mfcc = mfcc.unsqueeze(1)

        if lfcc.dim() == 3:
            lfcc = lfcc.unsqueeze(1)

        if self.freeze_backbones:
            with torch.no_grad():
                emb_mfcc = self.mfcc_net(mfcc)
                emb_lfcc = self.lfcc_net(lfcc)
        else:
            emb_mfcc = self.mfcc_net(mfcc)
            emb_lfcc = self.lfcc_net(lfcc)

        t_mfcc = self.mfcc_transform(emb_mfcc)
        t_lfcc = self.lfcc_transform(emb_lfcc)

        diff = torch.abs(t_mfcc - t_lfcc)
        interaction = t_mfcc * t_lfcc

        # Boundary-aware waveform representation
        wave_emb = self.wave_net(waveform)

        fused = torch.cat(
            (
                t_mfcc,
                t_lfcc,
                diff,
                interaction,
                wave_emb
            ),
            dim=1
        )

        features = self.fusion(fused)
        return self.classifier(features, label)


# ==========================================
# BENCHMARKING UTILITY
# ==========================================
def measure_computational_cost(
    model,
    input_size=(1, 400, 64),
    device='cuda'
):
    """
    Measures parameters, inference latency, and VRAM usage for a given model.
    Assumes models take (mfcc, lfcc) as input.
    """

    model = model.to(device)
    model.eval()

    # 1. Measure Parameters
    num_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    # Create dummy inputs
    dummy_mfcc = torch.randn(input_size).to(device)
    dummy_lfcc = torch.randn(input_size).to(device)

    # 2. Measure VRAM Usage
    vram_used = 0

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

        with torch.no_grad():
            _ = model(dummy_mfcc, dummy_lfcc)

        vram_used = (
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 2)
        )

    # 3. Measure Latency
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_mfcc, dummy_lfcc)

    iterations = 100

    if device == 'cuda':
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        starter.record()

        with torch.no_grad():
            for _ in range(iterations):
                _ = model(dummy_mfcc, dummy_lfcc)

        ender.record()
        torch.cuda.synchronize()

        total_time = (
            starter.elapsed_time(ender) / 1000.0
        )
    else:
        start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(iterations):
                _ = model(dummy_mfcc, dummy_lfcc)

        total_time = time.perf_counter() - start_time

    avg_latency = (
        total_time / iterations
    ) * 1000

    print(f"--- Model: {model.__class__.__name__} ---")
    print(f"Parameters: {num_params / 1e6:.2f} M")
    print(f"VRAM Usage: {vram_used:.2f} MB (forward pass peak)")
    print(
        f"Latency:    {avg_latency:.2f} ms per batch "
        f"(batch_size={input_size[0]})\n"
    )

    return {
        "parameters_M": num_params / 1e6,
        "vram_MB": vram_used,
        "latency_ms": avg_latency
    }