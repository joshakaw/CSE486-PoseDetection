"""
Run this BEFORE training to find exactly where NaN enters.
It checks: raw data -> normalization -> encoder -> residual -> decoder -> heatmaps -> gt_maps
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ── Copy your dataset class here exactly ─────────────────────────────────────
class MMVRDataset(Dataset):
    def __init__(self, root, max_samples=200):
        self.samples = []
        for file in os.listdir(root):
            if file.endswith("_radar.npz"):
                idx = file.replace("_radar.npz", "")
                base = os.path.join(root, idx)
                self.samples.append(base)
        self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        base = self.samples[i]
        radar = np.load(base + "_radar.npz")
        hori = radar["hm_hori"].astype(np.float32)
        vert = radar["hm_vert"].astype(np.float32)
        pose = np.load(base + "_pose.npz")
        kp = pose["kp"][0]

        # hm_vert is all-NaN across the dataset; replace before any arithmetic
        hori = np.nan_to_num(hori, nan=0.0, posinf=0.0, neginf=0.0)
        vert = np.nan_to_num(vert, nan=0.0, posinf=0.0, neginf=0.0)

        # hm_hori has values in 10M-5B range; clip to p99 then normalize
        hori = np.clip(hori, 0.0, np.percentile(hori, 99))
        vert = np.clip(vert, 0.0, np.percentile(vert, 99))

        # parentheses required: division before addition otherwise
        hori = hori / (np.max(hori) + 1e-6)
        vert = vert / (np.max(vert) + 1e-6)

        radar_input = np.stack([hori, vert], axis=0)

        kp = kp.copy().astype(np.float32)
        # kp x maxes at ~260 → source width is 320, not 640
        kp[:, 0] *= 128 / 320
        kp[:, 1] *= 256 / 480

        return (
            torch.tensor(radar_input, dtype=torch.float32),
            torch.tensor(kp, dtype=torch.float32)
        )


def check(name, t):
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    print(f"  {name:30s}  shape={str(tuple(t.shape)):20s}  "
          f"min={t.min().item():10.4f}  max={t.max().item():10.4f}  "
          f"NaN={has_nan}  Inf={has_inf}")
    return has_nan or has_inf


# ── 1. Inspect raw files ──────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1: Raw file inspection")
root = "./dataset"
files = [f for f in os.listdir(root) if f.endswith("_radar.npz")]
print(f"  Found {len(files)} radar files")

# Check first 3 samples
for fname in files[:3]:
    idx = fname.replace("_radar.npz", "")
    base = os.path.join(root, idx)
    radar = np.load(base + "_radar.npz")
    pose  = np.load(base + "_pose.npz")

    hori = radar["hm_hori"]
    vert = radar["hm_vert"]
    kp   = pose["kp"][0]

    print(f"\n  Sample: {idx}")
    print(f"    hm_hori shape : {hori.shape}  dtype={hori.dtype}")
    print(f"    hm_vert shape : {vert.shape}  dtype={vert.dtype}")
    print(f"    hm_hori range : [{hori.min():.4f}, {hori.max():.4f}]  "
          f"NaN={np.isnan(hori).any()}  Inf={np.isinf(hori).any()}")
    print(f"    hm_vert range : [{vert.min():.4f}, {vert.max():.4f}]  "
          f"NaN={np.isnan(vert).any()}  Inf={np.isinf(vert).any()}")
    print(f"    kp shape      : {kp.shape}  dtype={kp.dtype}")
    print(f"    kp[:, 2] (vis): {kp[:, 2]}")   # visibility flags
    print(f"    kp x range    : [{kp[:,0].min():.1f}, {kp[:,0].max():.1f}]")
    print(f"    kp y range    : [{kp[:,1].min():.1f}, {kp[:,1].max():.1f}]")
    print(f"    pose keys     : {list(pose.keys())}")


# ── 2. Dataset __getitem__ output ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Dataset output (after scaling)")
dataset = MMVRDataset(root, max_samples=200)
print(f"  Dataset length: {len(dataset)}")

bad_samples = []
for i in range(min(len(dataset), 50)):
    x, kp = dataset[i]
    if torch.isnan(x).any() or torch.isnan(kp).any():
        bad_samples.append(i)

print(f"  Samples with NaN in first 50: {bad_samples}")

x0, kp0 = dataset[0]
print(f"\n  Sample 0:")
check("radar_input", x0)
check("keypoints", kp0)
print(f"  kp x range (scaled): [{kp0[:,0].min():.2f}, {kp0[:,0].max():.2f}]  "
      f"(should be ~[0, 128])")
print(f"  kp y range (scaled): [{kp0[:,1].min():.2f}, {kp0[:,1].max():.2f}]  "
      f"(should be ~[0, 256])")


# ── 3. GT heatmap generation ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: GT heatmap generation")

VIS_THRESH = 0.3  # visibility is float [0,1]; v > 0 catches near-invisible kps

def make_heatmaps(kp, H=256, W=128, sigma=4):
    maps = np.zeros((17, H, W), dtype=np.float32)
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    for i in range(17):
        x, y, v = kp[i]
        if v > VIS_THRESH:
            maps[i] = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    return torch.tensor(maps, dtype=torch.float32)

gt = make_heatmaps(kp0.numpy())
bad = check("gt_maps sample 0", gt)
if bad:
    print("  !! NaN/Inf in GT heatmaps — likely a kp coordinate issue")
    print(f"  kp0 raw:\n{kp0.numpy()}")


# ── 4. Forward pass layer by layer ───────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: Model forward pass (layer by layer)")

class SoftArgmax2D(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        x_flat = F.softmax(x_flat, dim=-1)
        idx = torch.arange(H * W, device=x.device).float()
        xs = idx % W
        ys = torch.div(idx, W, rounding_mode='floor')
        x_coord = torch.sum(x_flat * xs, dim=-1)
        y_coord = torch.sum(x_flat * ys, dim=-1)
        return torch.stack([x_coord, y_coord], dim=-1)

class PoseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.res = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64)
        )
        self.dec = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.out = nn.Conv2d(32, 17, 1)
        self.soft_argmax = SoftArgmax2D()

    def forward_debug(self, x):
        print("  Input          :", end=" "); check("", x)

        x = self.enc(x)
        print("  After enc      :", end=" "); bad = check("", x)
        if bad: return None, None

        res = self.res(x)
        print("  After res block:", end=" "); check("", res)
        x = F.relu(x + res)
        print("  After res+relu :", end=" "); bad = check("", x)
        if bad: return None, None

        x = self.dec(x)
        print("  After dec      :", end=" "); bad = check("", x)
        if bad: return None, None

        heatmaps = self.out(x)
        print("  After out conv :", end=" "); bad = check("", heatmaps)
        if bad: return None, None

        coords = self.soft_argmax(heatmaps)
        print("  After soft_argmax:", end=" "); check("", coords)

        return heatmaps, coords

model = PoseCNN()
model.eval()

loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True)
x_batch, kp_batch = next(iter(loader))
print(f"\n  Batch shape: {x_batch.shape}")

with torch.no_grad():
    heatmaps, coords = model.forward_debug(x_batch)

if heatmaps is None:
    print("\n  !! NaN detected in forward pass above — "
          "the layer printed just before this is the culprit")
else:
    gt_batch = torch.stack(
        [make_heatmaps(k.numpy()) for k in kp_batch]
    )
    loss = nn.MSELoss()(heatmaps, gt_batch)
    print(f"\n  Loss on first batch (no grad): {loss.item():.6f}")
    print(f"  Loss NaN: {torch.isnan(loss).item()}")

print("\n" + "=" * 70)
print("STEP 5: Check if any radar file has zero max (causes div-by-zero)")
zero_max = []
for i, base in enumerate(dataset.samples):
    radar = np.load(base + "_radar.npz")
    hori = radar["hm_hori"]
    vert = radar["hm_vert"]
    stacked = np.stack([hori, vert])
    if np.max(stacked) < 1e-8:
        zero_max.append(i)
print(f"  Samples with near-zero max (bad normalization): {zero_max}")