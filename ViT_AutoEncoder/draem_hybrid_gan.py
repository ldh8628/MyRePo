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

from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report


# ============================================================
# 1. Dataset (정상 데이터만 사용)
# ============================================================

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


# ============================================================
# 2. CycleGAN Generator (cyclegan_train_infer.py 와 동일 구조)
# ============================================================

class ResidualBlock_GAN(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class CycleGANGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual=9):
        super().__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(n_residual):
            model += [ResidualBlock_GAN(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2,
                                   padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def load_cyclegan_generator(gan_ckpt_path, device):
    """
    cyclegan_train_infer.py 에서 저장한 G_AB.pth 를 로드.
    gan_ckpt_path 가 없으면 None 리턴 (GAN 미사용).
    """
    if gan_ckpt_path is None or gan_ckpt_path == "" or (not os.path.isfile(gan_ckpt_path)):
        print(f"[INFO] GAN checkpoint not found or not given. GAN anomaly OFF.")
        return None

    netG = CycleGANGenerator().to(device)
    state = torch.load(gan_ckpt_path, map_location=device)
    netG.load_state_dict(state)
    netG.eval()
    for p in netG.parameters():
        p.requires_grad = False
    print(f"[INFO] Loaded CycleGAN G_AB from {gan_ckpt_path}")
    return netG


# ============================================================
# 3. DRAEM Networks (ReconstructionNet, SegmentNet)
# ============================================================

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
        return torch.tanh(self.unet(x))  # [-1, 1]


class SegmentNet(nn.Module):
    """
    Input : concat( corrupted, reconstructed ) => (B, 6, H, W)
    Output: anomaly mask logit (B, 1, H, W)
    """
    def __init__(self):
        super().__init__()
        self.unet = UNet(in_ch=6, out_ch=1, base_ch=16)

    def forward(self, x):
        return self.unet(x)


# ============================================================
# 4. 랜덤 synthetic anomaly + GAN anomaly
# ============================================================

def generate_random_anomaly(x, min_ratio=0.01, max_ratio=0.20, strength=1.0):
    """
    기본 DRAEM 스타일 랜덤 blob anomaly.
    x: (B,3,H,W), [-1,1]
    min_ratio, max_ratio: 전체 픽셀 대비 anomaly 비율 범위.
    """
    B, C, H, W = x.shape
    device = x.device

    # coarse noise
    noise = torch.rand(B, 1, H // 4, W // 4, device=device)
    noise = nn.functional.interpolate(noise, size=(H, W),
                                      mode="bilinear",
                                      align_corners=False)

    mask = torch.zeros_like(noise)

    for b in range(B):
        flat = noise[b, 0].view(-1)
        numel = flat.numel()
        target_ratio = float(np.random.uniform(min_ratio, max_ratio))
        k = max(1, int(numel * target_ratio))
        topk_vals, _ = torch.topk(flat, k)
        thr = topk_vals[-1]
        mb = (noise[b, 0] >= thr).float().view(1, H, W)

        # dilation으로 blob을 조금 키움
        mb = nn.functional.max_pool2d(mb, kernel_size=3, stride=1, padding=1)
        mb = (mb > 0).float()

        mask[b] = mb

    # 랜덤 RGB 노이즈 [-1,1]
    anomaly_tex = (torch.rand_like(x) * 2.0 - 1.0) * strength

    x_anom = x * (1.0 - mask) + anomaly_tex * mask
    return x_anom, mask


def generate_gan_based_anomaly(
    x_normal,
    netG,
    min_ratio=0.01,
    max_ratio=0.20,
    dilate=True
):
    """
    x_normal: (B,3,H,W), [-1,1]
    netG   : CycleGANGenerator (normal -> defect)
    min_ratio, max_ratio: anomaly 비율 제약 (0~1).
    dilate: True이면 max_pool2d 로 blob을 조금 키움.

    return:
        x_anom: (B,3,H,W)
        mask  : (B,1,H,W) in {0,1}
    """
    device = x_normal.device
    B, C, H, W = x_normal.shape

    with torch.no_grad():
        x_defect = netG(x_normal)  # [-1,1]

    diff = torch.mean(torch.abs(x_defect - x_normal), dim=1, keepdim=True)  # (B,1,H,W)
    mask = torch.zeros_like(diff)

    for b in range(B):
        flat = diff[b, 0].view(-1)
        numel = flat.numel()

        target_ratio = float(np.random.uniform(min_ratio, max_ratio))
        k = max(1, int(numel * target_ratio))

        topk_vals, _ = torch.topk(flat, k)
        thr = topk_vals[-1]

        mb = (diff[b, 0] >= thr).float().view(1, H, W)

        if dilate:
            mb = nn.functional.max_pool2d(mb, kernel_size=3, stride=1, padding=1)
            mb = (mb > 0).float()

        mask[b] = mb

    x_anom = x_normal * (1.0 - mask) + x_defect * mask
    return x_anom, mask


# ============================================================
# 5. Training (랜덤 + GAN 혼합 DRAEM)
# ============================================================

def train_draem_hybrid(opt):
    import numpy as np  # 배치마다 GAN 사용 여부를 위한 random

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(opt.ckpt_dir, exist_ok=True)

    dataset = NormalDataset(opt.normal_dir, img_size=opt.img_size)
    loader = DataLoader(dataset,
                        batch_size=opt.batch_size,
                        shuffle=True,
                        num_workers=opt.num_workers,
                        drop_last=True)

    netG = load_cyclegan_generator(opt.gan_ckpt, device)

    recon_net = ReconstructionNet().to(device)
    seg_net = SegmentNet().to(device)

    # (옵션) 기본 DRAEM 체크포인트에서 초기화
    if opt.base_ckpt is not None and opt.base_ckpt != "" and os.path.isfile(opt.base_ckpt):
        print(f"[INFO] Loading base DRAEM checkpoint: {opt.base_ckpt}")
        base = torch.load(opt.base_ckpt, map_location=device)
        recon_net.load_state_dict(base["recon_net"])
        seg_net.load_state_dict(base["seg_net"])

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

            use_gan = (netG is not None) and (np.random.rand() < opt.gan_prob)

            if use_gan:
                x_anom, mask = generate_gan_based_anomaly(
                    x_normal,
                    netG,
                    min_ratio=opt.gan_min_ratio,
                    max_ratio=opt.gan_max_ratio,
                    dilate=True
                )
            else:
                x_anom, mask = generate_random_anomaly(
                    x_normal,
                    min_ratio=opt.rand_min_ratio,
                    max_ratio=opt.rand_max_ratio,
                    strength=opt.rand_strength
                )

            x_recon = recon_net(x_anom)
            seg_input = torch.cat([x_anom, x_recon], dim=1)
            mask_logit = seg_net(seg_input)

            loss_recon = recon_criterion(x_recon, x_normal)
            loss_seg = seg_criterion(mask_logit, mask)

            loss = opt.lambda_recon * loss_recon + opt.lambda_seg * loss_seg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(
                loss=float(loss.item()),
                recon=float(loss_recon.item()),
                seg=float(loss_seg.item()),
                use_gan=int(use_gan)
            )

        avg_loss = epoch_loss / len(loader)
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.6f}")

        ckpt_path = os.path.join(opt.ckpt_dir, f"draem_hybrid_epoch_{epoch+1}.pth")
        torch.save({
            "recon_net": recon_net.state_dict(),
            "seg_net": seg_net.state_dict(),
            "epoch": epoch + 1
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")


# ============================================================
# 6. Inference (heatmap / overlay 저장)
# ============================================================

def infer_draem(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(opt.out_dir, exist_ok=True)

    ckpts = sorted(glob(os.path.join(opt.ckpt_dir, "draem_hybrid_epoch_*.pth")))
    if len(ckpts) == 0:
        raise FileNotFoundError("No checkpoint found in ckpt_dir.")
    ckpt_path = ckpts[-1]
    print(f"[INFO] Loading checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    recon_net = ReconstructionNet().to(device)
    seg_net = SegmentNet().to(device)
    recon_net.load_state_dict(checkpoint["recon_net"])
    seg_net.load_state_dict(checkpoint["seg_net"])

    recon_net.eval()
    seg_net.eval()

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

            x_recon = recon_net(x)
            seg_input = torch.cat([x, x_recon], dim=1)
            mask_logit = seg_net(seg_input)
            mask_prob = torch.sigmoid(mask_logit)[0, 0].cpu().numpy()

            heat = (mask_prob * 255).astype(np.uint8)
            heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
            heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

            img_np = np.array(img_pil.resize((opt.img_size, opt.img_size)))
            overlay = cv2.addWeighted(img_np, 0.6, heat_color, 0.4, 0)

            fname = os.path.basename(p)
            base_name, _ = os.path.splitext(fname)
            out_heat = os.path.join(opt.out_dir, base_name + "_heat.png")
            out_overlay = os.path.join(opt.out_dir, base_name + "_overlay.png")

            cv2.imwrite(out_heat, cv2.cvtColor(heat_color, cv2.COLOR_RGB2BGR))
            cv2.imwrite(out_overlay, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print("Inference finished.")


# ============================================================
# 7. 평가용 anomaly score 계산 + 시각화 저장
# ============================================================

def compute_anomaly_score_and_save(
    img_path, label_name, recon_net, seg_net, transform,
    device, img_size, save_root
):
    img_pil = Image.open(img_path).convert("RGB")
    x = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        x_recon = recon_net(x)
        seg_input = torch.cat([x, x_recon], dim=1)
        mask_logit = seg_net(seg_input)
        mask_prob = torch.sigmoid(mask_logit)[0, 0].cpu().numpy()

    score = float(mask_prob.max())

    heat = (mask_prob * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    img_np = np.array(img_pil.resize((img_size, img_size)))
    overlay = cv2.addWeighted(img_np, 0.6, heat_color, 0.4, 0)

    label_dir = os.path.join(save_root, label_name)
    os.makedirs(label_dir, exist_ok=True)

    fname = os.path.basename(img_path)
    base_name, _ = os.path.splitext(fname)
    out_heat = os.path.join(label_dir, base_name + "_heat.png")
    out_overlay = os.path.join(label_dir, base_name + "_overlay.png")

    cv2.imwrite(out_heat, cv2.cvtColor(heat_color, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_overlay, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return score


def eval_draem_image_level(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(opt.eval_out_dir, exist_ok=True)

    ckpts = sorted(glob(os.path.join(opt.ckpt_dir, "draem_hybrid_epoch_*.pth")))
    if len(ckpts) == 0:
        raise FileNotFoundError("No checkpoint found in ckpt_dir.")
    ckpt_path = ckpts[-1]
    print(f"[INFO] Loading checkpoint for eval: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    recon_net = ReconstructionNet().to(device)
    seg_net = SegmentNet().to(device)
    recon_net.load_state_dict(checkpoint["recon_net"])
    seg_net.load_state_dict(checkpoint["seg_net"])
    recon_net.eval()
    seg_net.eval()

    transform = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])

    def get_paths(root):
        exts = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")
        ps = []
        for e in exts:
            ps.extend(glob(os.path.join(root, f"*.{e}")))
        return sorted(ps)

    normal_paths = get_paths(opt.normal_dir)
    defect_paths = get_paths(opt.defect_dir)
    print(f"[INFO] #normal={len(normal_paths)}, #defect={len(defect_paths)}")

    y_true = []
    y_score = []
    filenames = []

    for p in tqdm(normal_paths, desc="Eval normal"):
        score = compute_anomaly_score_and_save(
            img_path=p,
            label_name="normal",
            recon_net=recon_net,
            seg_net=seg_net,
            transform=transform,
            device=device,
            img_size=opt.img_size,
            save_root=opt.eval_out_dir
        )
        y_true.append(0)
        y_score.append(score)
        filenames.append(os.path.basename(p))

    for p in tqdm(defect_paths, desc="Eval defect"):
        score = compute_anomaly_score_and_save(
            img_path=p,
            label_name="defect",
            recon_net=recon_net,
            seg_net=seg_net,
            transform=transform,
            device=device,
            img_size=opt.img_size,
            save_root=opt.eval_out_dir
        )
        y_true.append(1)
        y_score.append(score)
        filenames.append(os.path.basename(p))

    y_true = np.array(y_true, dtype=np.int32)
    y_score = np.array(y_score, dtype=np.float32)

    auroc = roc_auc_score(y_true, y_score)
    print(f"\n[RESULT] Image-level AUROC: {auroc:.4f}")

    thr = float(np.median(y_score))
    print(f"[INFO] Threshold (median of scores): {thr:.4f}")
    y_pred = (y_score >= thr).astype(np.int32)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true, y_pred,
        target_names=["normal", "defect"],
        digits=4
    )

    print("\n[RESULT] Confusion Matrix (rows:true, cols:pred) [0=normal,1=defect]:")
    print(cm)
    print("\n[RESULT] Classification report:")
    print(report)

    metrics_path = os.path.join(opt.eval_out_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Image-level AUROC: {auroc:.6f}\n")
        f.write(f"Threshold (median): {thr:.6f}\n\n")
        f.write("Confusion Matrix (rows:true, cols:pred) [0=normal,1=defect]:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification report:\n")
        f.write(report)
    print(f"[INFO] Saved metrics to {metrics_path}")

    csv_path = os.path.join(opt.eval_out_dir, "scores.csv")
    with open(csv_path, "w") as f:
        f.write("filename,label,score,pred\n")
        for fn, y, s, yp in zip(filenames, y_true, y_score, y_pred):
            f.write(f"{fn},{y},{s:.6f},{int(yp)}\n")
    print(f"[INFO] Saved scores to {csv_path}")


# ============================================================
# 8. Main / Argparse
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid DRAEM (random + GAN-based synthetic anomaly)"
    )

    # data
    parser.add_argument("--normal_dir", type=str, default="./data/normal")
    parser.add_argument("--defect_dir", type=str, default="./data/defect")
    parser.add_argument("--img_size", type=int, default=256)

    # GAN
    parser.add_argument("--gan_ckpt", type=str, default="./G_AB.pth",
                        help="CycleGAN G_AB checkpoint (없으면 빈 문자열 또는 존재하지 않는 경로)")
    parser.add_argument("--gan_prob", type=float, default=0.5,
                        help="배치당 GAN anomaly 사용할 확률 (0~1)")
    parser.add_argument("--gan_min_ratio", type=float, default=0.02)
    parser.add_argument("--gan_max_ratio", type=float, default=0.15)

    # Random anomaly
    parser.add_argument("--rand_min_ratio", type=float, default=0.02)
    parser.add_argument("--rand_max_ratio", type=float, default=0.15)
    parser.add_argument("--rand_strength", type=float, default=1.0)

    # training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts_draem_hybrid")

    # (옵션) 기본 DRAEM checkpoint 로부터 초기화
    parser.add_argument("--base_ckpt", type=str, default="",
                        help="기본 DRAEM checkpoint 경로 (없으면 빈 문자열)")

    # loss weights
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_seg", type=float, default=1.0)

    # inference / eval
    parser.add_argument("--in_dir", type=str, default="./data/defect")
    parser.add_argument("--out_dir", type=str, default="./outputs_draem_hybrid")
    parser.add_argument("--eval_out_dir", type=str, default="./eval_draem_hybrid")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--eval", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()

    if opt.train:
        train_draem_hybrid(opt)
    elif opt.infer:
        infer_draem(opt)
    elif opt.eval:
        eval_draem_image_level(opt)
    else:
        print("Specify --train or --infer or --eval")
