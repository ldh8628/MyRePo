import os
import argparse
from glob import glob

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


# =========================
# 1. Dataset
# =========================

class NormalDataset(Dataset):
    """
    Uses only normal images for training DRAEM.
    Folder structure: ./data/normal/*.jpg|*.png|...
    """
    def __init__(self, root, img_size=256):
        self.files = []
        exts = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")
        for e in exts:
            self.files.extend(glob(os.path.join(root, f"*.{e}")))
        self.files = sorted(self.files)

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        img = self.transform(img)
        return img


# =========================
# 2. Synthetic Anomaly Generator
# =========================

def generate_synthetic_anomaly(x, anomaly_strength=1.0):
    """
    x: (B, 3, H, W), normalized in [-1, 1]
    returns:
        x_anom: (B, 3, H, W)
        mask:   (B, 1, H, W) in {0,1}
    This is a simple DRAEM-style synthetic anomaly using random blobs.
    """
    B, C, H, W = x.shape
    device = x.device

    # coarse random noise
    noise = torch.rand(B, 1, H // 4, W // 4, device=device)
    noise = torch.nn.functional.interpolate(noise, size=(H, W),
                                            mode="bilinear",
                                            align_corners=False)
    # threshold to get blobs
    thresh = 0.7
    mask = (noise > thresh).float()  # (B, 1, H, W)

    # ensure some blobs exist
    if mask.sum() == 0:
        # set one random square as anomaly
        for b in range(B):
            cy = torch.randint(0, H, (1,), device=device)
            cx = torch.randint(0, W, (1,), device=device)
            h = H // 8
            w = W // 8
            y1 = max(0, int(cy - h // 2))
            y2 = min(H, int(cy + h // 2))
            x1 = max(0, int(cx - w // 2))
            x2 = min(W, int(cx + w // 2))
            mask[b, 0, y1:y2, x1:x2] = 1.0

    # anomaly texture: random RGB noise in [-1, 1]
    anomaly_tex = (torch.rand_like(x) * 2.0 - 1.0) * anomaly_strength

    x_anom = x * (1.0 - mask) + anomaly_tex * mask
    return x_anom, mask


# =========================
# 3. DRAEM Networks
#    - ReconstructionNet: U-Net encoder-decoder
#    - SegmentNet: U-Net taking [x_anom, x_recon] -> anomaly mask
# =========================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_ch, out_ch),
            ConvBlock(out_ch, out_ch),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.block(x)
        x_down = self.pool(x)
        return x, x_down


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            ConvBlock(in_ch, out_ch),
            ConvBlock(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # align shapes if needed
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffX != 0 or diffY != 0:
            x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        x = self.block(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch=32):
        super().__init__()
        self.d1 = DownBlock(in_ch, base_ch)
        self.d2 = DownBlock(base_ch, base_ch * 2)
        self.d3 = DownBlock(base_ch * 2, base_ch * 4)
        self.d4 = DownBlock(base_ch * 4, base_ch * 8)

        self.bottleneck = nn.Sequential(
            ConvBlock(base_ch * 8, base_ch * 16),
            ConvBlock(base_ch * 16, base_ch * 16),
        )

        self.u4 = UpBlock(base_ch * 16, base_ch * 8)
        self.u3 = UpBlock(base_ch * 8, base_ch * 4)
        self.u2 = UpBlock(base_ch * 4, base_ch * 2)
        self.u1 = UpBlock(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        s1, x1 = self.d1(x)
        s2, x2 = self.d2(x1)
        s3, x3 = self.d3(x2)
        s4, x4 = self.d4(x3)

        b = self.bottleneck(x4)

        x = self.u4(b, s4)
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)

        out = self.out_conv(x)
        return out


class ReconstructionNet(nn.Module):
    """
    Input : synthetic anomaly image (B, 3, H, W)
    Output: reconstructed normal image (B, 3, H, W)
    """
    def __init__(self):
        super().__init__()
        self.unet = UNet(in_ch=3, out_ch=3, base_ch=32)

    def forward(self, x):
        return torch.tanh(self.unet(x))  # keep in [-1, 1]


class SegmentNet(nn.Module):
    """
    Input : concat( corrupted, reconstructed ) => (B, 6, H, W)
    Output: anomaly mask logit (B, 1, H, W)
    """
    def __init__(self):
        super().__init__()
        self.unet = UNet(in_ch=6, out_ch=1, base_ch=16)

    def forward(self, x):
        return self.unet(x)  # logits (no sigmoid)


# =========================
# 4. Training
# =========================

def train_draem(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(opt.ckpt_dir, exist_ok=True)

    dataset = NormalDataset(opt.normal_dir, img_size=opt.img_size)
    loader = DataLoader(dataset,
                        batch_size=opt.batch_size,
                        shuffle=True,
                        num_workers=opt.num_workers,
                        drop_last=True)

    recon_net = ReconstructionNet().to(device)
    seg_net = SegmentNet().to(device)

    recon_criterion = nn.L1Loss()
    seg_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        list(recon_net.parameters()) + list(seg_net.parameters()),
        lr=opt.lr,
        betas=(0.5, 0.999),
        weight_decay=opt.weight_decay
    )

    for epoch in range(opt.epochs):
        recon_net.train()
        seg_net.train()
        epoch_loss = 0.0

        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{opt.epochs}")
        for x_normal in loop:
            x_normal = x_normal.to(device)

            # synthetic anomaly
            x_anom, mask = generate_synthetic_anomaly(
                x_normal, anomaly_strength=opt.anomaly_strength
            )
            mask = mask.to(device)

            # reconstruction
            x_recon = recon_net(x_anom)

            # segmentation input: concat(anom, recon)
            seg_input = torch.cat([x_anom, x_recon], dim=1)
            mask_logit = seg_net(seg_input)

            # losses
            loss_recon = recon_criterion(x_recon, x_normal)
            loss_seg = seg_criterion(mask_logit, mask)

            loss = opt.lambda_recon * loss_recon + opt.lambda_seg * loss_seg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item(),
                             recon=loss_recon.item(),
                             seg=loss_seg.item())

        avg_loss = epoch_loss / len(loader)
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.6f}")

        ckpt_path = os.path.join(opt.ckpt_dir, f"draem_epoch_{epoch+1}.pth")
        torch.save({
            "recon_net": recon_net.state_dict(),
            "seg_net": seg_net.state_dict(),
            "epoch": epoch + 1
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")


# =========================
# 5. Inference (heatmap / mask save)
# =========================

def infer_draem(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(opt.out_dir, exist_ok=True)

    # load latest checkpoint
    ckpts = sorted(glob(os.path.join(opt.ckpt_dir, "draem_epoch_*.pth")))
    if len(ckpts) == 0:
        raise FileNotFoundError("No checkpoint found in ckpt_dir.")
    ckpt_path = ckpts[-1]
    print(f"Loading checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    recon_net = ReconstructionNet().to(device)
    seg_net = SegmentNet().to(device)
    recon_net.load_state_dict(checkpoint["recon_net"])
    seg_net.load_state_dict(checkpoint["seg_net"])

    recon_net.eval()
    seg_net.eval()

    # image list
    exts = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")
    img_paths = []
    for e in exts:
        img_paths.extend(glob(os.path.join(opt.in_dir, f"*.{e}")))
    img_paths = sorted(img_paths)

    transform = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])

    with torch.no_grad():
        for p in tqdm(img_paths, desc="Inference"):
            img_pil = Image.open(p).convert("RGB")
            x = transform(img_pil).unsqueeze(0).to(device)

            # reconstruction & anomaly map
            # (여기서는 실제 anomaly가 들어온다고 가정)
            # segmentation 입력: concat(x, recon)
            x_recon = recon_net(x)
            seg_input = torch.cat([x, x_recon], dim=1)
            mask_logit = seg_net(seg_input)
            mask_prob = torch.sigmoid(mask_logit)[0, 0].cpu().numpy()

            # to heatmap image (0~255)
            heat = (mask_prob * 255).astype(np.uint8)
            heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
            heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

            # original image (0~255)
            img_np = np.array(img_pil.resize((opt.img_size, opt.img_size)))
            # overlay
            overlay = cv2.addWeighted(img_np, 0.6, heat_color, 0.4, 0)

            fname = os.path.basename(p)
            out_heat = os.path.join(opt.out_dir, fname.replace(".", "_heat."))
            out_overlay = os.path.join(opt.out_dir, fname.replace(".", "_overlay."))

            # choose png as default
            base_name = os.path.splitext(fname)[0]
            out_heat = os.path.join(opt.out_dir, base_name + "_heat.png")
            out_overlay = os.path.join(opt.out_dir, base_name + "_overlay.png")

            cv2.imwrite(out_heat, cv2.cvtColor(heat_color, cv2.COLOR_RGB2BGR))
            cv2.imwrite(out_overlay, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print("Inference finished.")


# =========================
# 6. Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="Basic DRAEM implementation")

    # data
    parser.add_argument("--normal_dir", type=str, default="./data/normal",
                        help="path to normal images (for training)")
    parser.add_argument("--defect_dir", type=str, default="./data/defect",
                        help="path to defect images (currently unused)")
    parser.add_argument("--img_size", type=int, default=256)

    # training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts_draem")

    # losses
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_seg", type=float, default=1.0)
    parser.add_argument("--anomaly_strength", type=float, default=1.0)

    # inference
    parser.add_argument("--in_dir", type=str, default="./data/defect",
                        help="input directory for inference")
    parser.add_argument("--out_dir", type=str, default="./outputs_draem")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--infer", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()

    if opt.train:
        train_draem(opt)
    elif opt.infer:
        infer_draem(opt)
    else:
        print("Specify either --train or --infer")
