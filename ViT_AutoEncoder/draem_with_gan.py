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
    """
    if not os.path.isfile(gan_ckpt_path):
        raise FileNotFoundError(f"GAN checkpoint not found: {gan_ckpt_path}")

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
# 4. GAN 기반 synthetic anomaly 생성
# ============================================================

def generate_gan_based_anomaly(x_normal, netG, diff_threshold=0.3, min_blob_ratio=0.01):
    """
    x_normal: (B,3,H,W), [-1,1]
    netG   : CycleGANGenerator (normal -> defect)
    diff_threshold: |G(x) - x| 평균 값이 이 이상인 픽셀을 anomaly 로 간주
    min_blob_ratio: mask 가 전체 픽셀 대비 이 비율보다 작으면 랜덤 영역 추가

    return:
        x_anom: (B,3,H,W)
        mask  : (B,1,H,W) in {0,1}
    """
    device = x_normal.device
    B, C, H, W = x_normal.shape

    with torch.no_grad():
        x_defect = netG(x_normal)  # [-1,1]

    # 채널 평균 L1 difference
    diff = torch.mean(torch.abs(x_defect - x_normal), dim=1, keepdim=True)  # (B,1,H,W)

    # threshold 로 mask 생성
    mask = (diff > diff_threshold).float()  # (B,1,H,W)

    # mask 가 너무 작은 경우 방어 코드
    for b in range(B):
        ratio = mask[b].mean().item()
        if ratio < min_blob_ratio:
            # 작은 랜덤 사각형 추가
            cy = torch.randint(0, H, (1,), device=device).item()
            cx = torch.randint(0, W, (1,), device=device).item()
            h = max(4, H // 16)
            w = max(4, W // 16)
            y1 = max(0, cy - h // 2)
            y2 = min(H, cy + h // 2)
            x1 = max(0, cx - w // 2)
            x2 = min(W, cx + w // 2)
            mask[b, 0, y1:y2, x1:x2] = 1.0

    # mask 영역만 GAN 결과를 섞어서 synthetic anomaly 생성
    x_anom = x_normal * (1.0 - mask) + x_defect * mask

    return x_anom, mask


# ============================================================
# 5. Training (DRAEM + CycleGAN 기반 synthetic anomaly)
# ============================================================

def train_draem_with_gan(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(opt.ckpt_dir, exist_ok=True)

    # Normal data loader
    dataset = NormalDataset(opt.normal_dir, img_size=opt.img_size)
    loader = DataLoader(dataset,
                        batch_size=opt.batch_size,
                        shuffle=True,
                        num_workers=opt.num_workers,
                        drop_last=True)

    # CycleGAN Generator (normal -> defect)
    netG = load_cyclegan_generator(opt.gan_ckpt, device)

    # DRAEM networks
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

            # GAN 기반 synthetic anomaly 생성
            x_anom, mask = generate_gan_based_anomaly(
                x_normal,
                netG,
                diff_threshold=opt.diff_threshold,
                min_blob_ratio=opt.min_blob_ratio
            )

            # reconstruction
            x_recon = recon_net(x_anom)

            # segmentation input: concat(anom, recon)
            seg_input = torch.cat([x_anom, x_recon], dim=1)
            mask_logit = seg_net(seg_input)

            # 손실
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

        ckpt_path = os.path.join(opt.ckpt_dir, f"draem_gan_epoch_{epoch+1}.pth")
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

    ckpts = sorted(glob(os.path.join(opt.ckpt_dir, "draem_gan_epoch_*.pth")))
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
    """
    각 이미지에 대해 anomaly score 계산 + heatmap/overlay 저장.
    label_name: "normal" 또는 "defect"
    save_root/label_name 폴더에 이미지 저장.
    """
    img_pil = Image.open(img_path).convert("RGB")
    x = transform(img_pil).unsqueeze(0).to(device)  # (1,3,H,W)

    with torch.no_grad():
        x_recon = recon_net(x)
        seg_input = torch.cat([x, x_recon], dim=1)
        mask_logit = seg_net(seg_input)
        mask_prob = torch.sigmoid(mask_logit)[0, 0].cpu().numpy()  # (H,W)

    # anomaly score: max prob
    score = float(mask_prob.max())

    # heatmap / overlay 저장
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
    """
    normal_dir, defect_dir 를 사용하여 이미지 단위 AUROC / confusion matrix 계산,
    동시에 eval_out_dir/normal, eval_out_dir/defect 에 heatmap/overlay 저장.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(opt.eval_out_dir, exist_ok=True)

    # ckpt 로드
    ckpts = sorted(glob(os.path.join(opt.ckpt_dir, "draem_gan_epoch_*.pth")))
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

    # 이미지 경로 수집
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

    # 정상: label 0
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

    # 불량: label 1
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

    # AUROC
    auroc = roc_auc_score(y_true, y_score)
    print(f"\n[RESULT] Image-level AUROC: {auroc:.4f}")

    # threshold (여기서는 median 사용; 필요하면 수정 가능)
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

    # metrics.txt 저장
    metrics_path = os.path.join(opt.eval_out_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Image-level AUROC: {auroc:.6f}\n")
        f.write(f"Threshold (median): {thr:.6f}\n\n")
        f.write("Confusion Matrix (rows:true, cols:pred) [0=normal,1=defect]:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification report:\n")
        f.write(report)
    print(f"[INFO] Saved metrics to {metrics_path}")

    # scores.csv 저장
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
        description="DRAEM with GAN-based synthetic anomaly (CycleGAN G_AB) + eval"
    )

    # data
    parser.add_argument("--normal_dir", type=str, default="./data/normal",
                        help="path to normal images")
    parser.add_argument("--defect_dir", type=str, default="./data/defect",
                        help="path to defect images")
    parser.add_argument("--img_size", type=int, default=256)

    # GAN
    parser.add_argument("--gan_ckpt", type=str, default="./G_AB.pth",
                        help="path to CycleGAN G_AB checkpoint")

    # training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts_draem_gan")

    # loss weights
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_seg", type=float, default=1.0)

    # GAN diff -> mask 설정
    parser.add_argument("--diff_threshold", type=float, default=0.3,
                        help="|G(x)-x| 채널 평균 값이 이 이상이면 anomaly 로 간주")
    parser.add_argument("--min_blob_ratio", type=float, default=0.01,
                        help="mask가 전체 픽셀 대비 이 비율보다 작으면 랜덤 영역 추가")

    # inference
    parser.add_argument("--in_dir", type=str, default="./data/defect",
                        help="input directory for inference")
    parser.add_argument("--out_dir", type=str, default="./outputs_draem_gan")

    # eval
    parser.add_argument("--eval_out_dir", type=str, default="./eval_draem_gan")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--eval", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()

    if opt.train:
        train_draem_with_gan(opt)
    elif opt.infer:
        infer_draem(opt)
    elif opt.eval:
        eval_draem_image_level(opt)
    else:
        print("Specify --train or --infer or --eval")
