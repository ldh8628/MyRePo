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

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report


# ============================================================
# 1. Dataset
# ============================================================

class NormalDataset(Dataset):
    """
    정상 이미지만 쓰는 Dataset.
    root/ 에 이미지들만 있다고 가정.
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


def get_image_paths(root):
    exts = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")
    paths = []
    for e in exts:
        paths.extend(glob(os.path.join(root, f"*.{e}")))
    return sorted(paths)


# ============================================================
# 2. CycleGAN Generator (G_AB) - 기존과 동일
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
# 3. DRAEM Networks
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
    def __init__(self):
        super().__init__()
        self.unet = UNet(in_ch=3, out_ch=3, base_ch=32)

    def forward(self, x):
        return torch.tanh(self.unet(x))


class SegmentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(in_ch=6, out_ch=1, base_ch=16)

    def forward(self, x):
        return self.unet(x)


# ============================================================
# 4. Synthetic anomaly (랜덤 + GAN)
# ============================================================

def generate_random_anomaly(x, min_ratio=0.02, max_ratio=0.15, strength=1.0):
    B, C, H, W = x.shape
    device = x.device

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

        mb = nn.functional.max_pool2d(mb, kernel_size=3, stride=1, padding=1)
        mb = (mb > 0).float()

        mask[b] = mb

    anomaly_tex = (torch.rand_like(x) * 2.0 - 1.0) * strength
    x_anom = x * (1.0 - mask) + anomaly_tex * mask
    return x_anom, mask


def generate_gan_based_anomaly(x_normal, netG,
                               min_ratio=0.02, max_ratio=0.15, dilate=True):
    if netG is None:
        # GAN이 없으면 그냥 랜덤 anomaly 로 대체
        return generate_random_anomaly(x_normal, min_ratio, max_ratio, strength=1.0)

    device = x_normal.device
    B, C, H, W = x_normal.shape

    with torch.no_grad():
        x_defect = netG(x_normal)

    diff = torch.mean(torch.abs(x_defect - x_normal), dim=1, keepdim=True)
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
# 5. Training (train_data/train/normal 사용)
# ============================================================

def train_draem_hybrid(opt):
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(opt.ckpt_dir, exist_ok=True)

    train_normal_dir = os.path.join(opt.data_root, "train", "normal")
    dataset = NormalDataset(train_normal_dir, img_size=opt.img_size)
    loader = DataLoader(dataset,
                        batch_size=opt.batch_size,
                        shuffle=True,
                        num_workers=opt.num_workers,
                        drop_last=True)

    netG = load_cyclegan_generator(opt.gan_ckpt, device)

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

    print(f"[INFO] #train normal = {len(dataset)}")

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
                    x_normal, netG,
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
# 6. 공통: score 계산 + ROC 기반 평가 (validation / test 공용)
# ============================================================

def load_latest_ckpt(ckpt_dir, device):
    ckpts = sorted(glob(os.path.join(ckpt_dir, "draem_hybrid_epoch_*.pth")))
    if len(ckpts) == 0:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    ckpt_path = ckpts[-1]
    print(f"[INFO] Load checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    recon_net = ReconstructionNet().to(device)
    seg_net = SegmentNet().to(device)
    recon_net.load_state_dict(checkpoint["recon_net"])
    seg_net.load_state_dict(checkpoint["seg_net"])
    recon_net.eval()
    seg_net.eval()
    return recon_net, seg_net, ckpt_path


def compute_score(img_path, recon_net, seg_net, transform, device, img_size):
    img_pil = Image.open(img_path).convert("RGB")
    x = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        x_recon = recon_net(x)
        seg_input = torch.cat([x, x_recon], dim=1)
        mask_logit = seg_net(seg_input)
        mask_prob = torch.sigmoid(mask_logit)[0, 0].cpu().numpy()

    score = float(mask_prob.max())
    return score


def eval_split(opt, split_name="validation"):
    """
    split_name: "validation" 또는 "test"
    data_root/split_name/normal, data_root/split_name/defect 사용
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(opt.eval_out_dir, exist_ok=True)

    recon_net, seg_net, ckpt_path = load_latest_ckpt(opt.ckpt_dir, device)

    transform = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])

    normal_dir = os.path.join(opt.data_root, split_name, "normal")
    defect_dir = os.path.join(opt.data_root, split_name, "defect")

    normal_paths = get_image_paths(normal_dir)
    defect_paths = get_image_paths(defect_dir)

    print(f"[INFO] [{split_name}] #normal={len(normal_paths)}, #defect={len(defect_paths)}")

    y_true = []
    y_score = []
    filenames = []

    for p in tqdm(normal_paths, desc=f"Eval {split_name} normal"):
        s = compute_score(p, recon_net, seg_net, transform, device, opt.img_size)
        y_true.append(0)
        y_score.append(s)
        filenames.append(os.path.basename(p))

    for p in tqdm(defect_paths, desc=f"Eval {split_name} defect"):
        s = compute_score(p, recon_net, seg_net, transform, device, opt.img_size)
        y_true.append(1)
        y_score.append(s)
        filenames.append(os.path.basename(p))

    y_true = np.array(y_true, dtype=np.int32)
    y_score = np.array(y_score, dtype=np.float32)

    auroc = roc_auc_score(y_true, y_score)
    print(f"\n[RESULT] [{split_name}] Image-level AUROC: {auroc:.6f}")

    fpr, tpr, thr = roc_curve(y_true, y_score)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_thr = float(thr[best_idx])
    print(f"[INFO] [{split_name}] Best threshold (Youden): {best_thr:.6f}")
    print(f"[INFO] [{split_name}] TPR={tpr[best_idx]:.4f}, FPR={fpr[best_idx]:.4f}")

    y_pred_best = (y_score >= best_thr).astype(np.int32)
    cm_best = confusion_matrix(y_true, y_pred_best, labels=[0, 1])
    report_best = classification_report(
        y_true, y_pred_best,
        target_names=["normal", "defect"],
        digits=4,
    )

    print(f"\n[RESULT] [{split_name}] Confusion Matrix @ best_thr:")
    print(cm_best)
    print(f"\n[RESULT] [{split_name}] Classification report @ best_thr:")
    print(report_best)

    metrics_path = os.path.join(opt.eval_out_dir, f"metrics_{split_name}.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"[{split_name}] Image-level AUROC: {auroc:.6f}\n\n")
        f.write(f"[{split_name}] Best threshold (Youden): {best_thr:.6f}\n")
        f.write(f"TPR={tpr[best_idx]:.6f}, FPR={fpr[best_idx]:.6f}\n\n")
        f.write("Confusion Matrix (rows:true, cols:pred) [0=normal,1=defect]:\n")
        f.write(str(cm_best) + "\n\n")
        f.write("Classification report:\n")
        f.write(report_best + "\n")
    print(f"[INFO] Saved metrics to {metrics_path}")

    csv_path = os.path.join(opt.eval_out_dir, f"scores_{split_name}.csv")
    with open(csv_path, "w") as f:
        f.write("filename,label,score,pred_best\n")
        for fn, y, s, yp in zip(filenames, y_true, y_score, y_pred_best):
            f.write(f"{fn},{y},{s:.6f},{int(yp)}\n")
    print(f"[INFO] Saved scores to {csv_path}")


# ============================================================
# 7. Main / Argparse
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="DRAEM hybrid with explicit train/validation/test folders"
    )

    parser.add_argument("--data_root", type=str, default="./train_data",
                        help="train_data root (contains train/ validation/ test/)")
    parser.add_argument("--img_size", type=int, default=256)

    # GAN
    parser.add_argument("--gan_ckpt", type=str, default="./G_AB.pth")
    parser.add_argument("--gan_prob", type=float, default=0.5)
    parser.add_argument("--gan_min_ratio", type=float, default=0.02)
    parser.add_argument("--gan_max_ratio", type=float, default=0.15)

    # Random anomaly
    parser.add_argument("--rand_min_ratio", type=float, default=0.02)
    parser.add_argument("--rand_max_ratio", type=float, default=0.15)
    parser.add_argument("--rand_strength", type=float, default=1.0)

    # training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts_draem_hybrid_split")

    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_seg", type=float, default=1.0)

    # eval
    parser.add_argument("--eval_out_dir", type=str, default="./eval_split")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval_val", action="store_true")
    parser.add_argument("--eval_test", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()

    if opt.train:
        train_draem_hybrid(opt)
    if opt.eval_val:
        eval_split(opt, split_name="validation")
    if opt.eval_test:
        eval_split(opt, split_name="test")
    if (not opt.train) and (not opt.eval_val) and (not opt.eval_test):
        print("Specify --train or --eval_val or --eval_test")
