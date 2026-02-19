import torch
import torch.nn as nn
import torch.nn.functional as F

class DCCEngine(nn.Module):
    def __init__(self, dim, grid_size=8):
        super().__init__()
        self.S = grid_size
        self.num_clusters = grid_size * grid_size

        # 🔧 Smaller initialization to prevent attention collapse
        self.centers = nn.Parameter(torch.randn(1, self.num_clusters, dim) * 0.02)

        # projection with normalization
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GroupNorm(num_groups=32, num_channels=dim)  # stable across resolutions
        )

        # Lightweight Mamba-style scanning on cluster grid
        self.cluster_scan = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.act = nn.SiLU()

    def forward(self, x):
        B, C, H, W = x.shape

        x_flat = x.flatten(2).transpose(1, 2)  # [B, N, C]
        centers = self.centers.expand(B, -1, -1)  # [B, K, C]

        sim = (x_flat @ centers.transpose(1, 2)) * (C ** -0.5)
        attn = sim.softmax(dim=-1)  # [B, N, K]

        cluster_feat = attn.transpose(1, 2) @ x_flat  # [B, K, C]
        S = self.S
        cluster_grid = cluster_feat.transpose(1, 2).reshape(B, C, S, S)

        cluster_refined = self.act(self.cluster_scan(cluster_grid))
        refined_flat = cluster_refined.flatten(2).transpose(1, 2)  # [B, K, C]
        out = attn @ refined_flat   # [B, N, C]
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return x + self.proj(out)

class SFABottleneck(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.freq_gate = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
        self.norm = nn.GroupNorm(num_groups=32, num_channels=dim)

    def forward(self, x):
        residual = x
        x = torch.clamp(x, min=-5.0, max=5.0)

        fft = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'))
        amp = torch.abs(fft)
        pha = torch.angle(fft)
        amp = torch.clamp(amp, min=0.0, max=100.0)

        mask = self.freq_gate(amp)
        refined_amp = amp * mask

        real_part = refined_amp * torch.cos(pha)
        imag_part = refined_amp * torch.sin(pha)
        fft_refined = torch.complex(real_part, imag_part)

        out = torch.fft.ifft2(torch.fft.ifftshift(fft_refined), norm='ortho').real
        out = torch.clamp(out, min=-5.0, max=5.0)

        # Add residual and normalize
        out = self.norm(out + residual)
        awareness = torch.mean(mask, dim=1, keepdim=True)
        return out, awareness

class TRIBridge(nn.Module):
    def __init__(self, dims):
        super().__init__()
        total_dim = sum(dims)

        self.integrity_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_dim, total_dim // 8, 1),
            nn.ReLU(),
            nn.Conv2d(total_dim // 8, total_dim, 1),
            nn.Sigmoid()
        )
        self.norm = nn.GroupNorm(num_groups=32, num_channels=total_dim)

        self.edge_conv = nn.Conv2d(dims[-1], 1, 3, padding=1)

    def forward(self, feats):
        target = feats[0].shape[-2:]
        aligned = [F.interpolate(f, size=target, mode='bilinear') for f in feats]
        cat = torch.cat(aligned, dim=1)

        weights = self.integrity_net(cat)
        refined = cat * weights
        refined = self.norm(refined)  # stabilize

        edge = torch.sigmoid(self.edge_conv(feats[-1]))
        return refined, edge

class SCMNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        dims = [64, 128, 256, 512]

        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], 3, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=dims[0])
        )

        # Encoder with DCC
        self.enc1 = DCCEngine(dims[0], grid_size=16)
        self.down1 = nn.Conv2d(dims[0], dims[1], 3, stride=2, padding=1)

        self.enc2 = DCCEngine(dims[1], grid_size=8)
        self.down2 = nn.Conv2d(dims[1], dims[2], 3, stride=2, padding=1)

        self.enc3 = DCCEngine(dims[2], grid_size=8)
        self.down3 = nn.Conv2d(dims[2], dims[3], 3, stride=2, padding=1)

        self.enc4 = DCCEngine(dims[3], grid_size=8)

        self.sfa = SFABottleneck(dims[3])
        self.tri = TRIBridge(dims)

        # Decoder with normalization
        self.dec3 = nn.Sequential(
            nn.Conv2d(dims[3] + dims[2] + 1, dims[2], 1),
            nn.GroupNorm(num_groups=32, num_channels=dims[2])
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(dims[2] + dims[1] + 1, dims[1], 1),
            nn.GroupNorm(num_groups=32, num_channels=dims[1])
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(dims[1] + dims[0] + 1, dims[0], 1),
            nn.GroupNorm(num_groups=16, num_channels=dims[0])
        )

        self.head = nn.Conv2d(dims[0], num_classes, 1)

    def forward(self, x):
        H, W = x.shape[-2:]

        x = self.stem(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))

        f, _ = self.sfa(e4)
        _, edge = self.tri([e1, e2, e3, e4])

        d3 = self.dec3(torch.cat([
            F.interpolate(f, size=e3.shape[-2:], mode='bilinear'),
            e3,
            F.interpolate(edge, size=e3.shape[-2:], mode='bilinear')
        ], dim=1))

        d2 = self.dec2(torch.cat([
            F.interpolate(d3, size=e2.shape[-2:], mode='bilinear'),
            e2,
            F.interpolate(edge, size=e2.shape[-2:], mode='bilinear')
        ], dim=1))

        d1 = self.dec1(torch.cat([
            F.interpolate(d2, size=e1.shape[-2:], mode='bilinear'),
            e1,
            F.interpolate(edge, size=e1.shape[-2:], mode='bilinear')
        ], dim=1))

        # ✅ NO CLAMP — architecture should be stable now
        return self.head(F.interpolate(d1, size=(H, W), mode='bilinear'))

if __name__ == "__main__":
    model = SCMNet(num_classes=7).cuda()
    x = torch.randn(1, 3, 512, 512).cuda()
    y = model(x)
    print("Output shape:", y.shape)
    print("Logits range: [{:.2f}, {:.2f}]".format(y.min().item(), y.max().item()))