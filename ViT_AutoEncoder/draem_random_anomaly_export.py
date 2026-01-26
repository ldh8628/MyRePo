import os
import argparse
from glob import glob

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


# ============================================================
# 1. Dataset (정상 이미지 전용)
# ============================================================

class NormalDataset(Dataset):
    """
    정상 이미지만 사용하는 Dataset.
    root/ 에 이미지 파일들만 있다고 가정.
    """
    def __init__(self, root, img_size=256):
        self.files = []
        exts = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")
        for e in exts:
            self.files.extend(glob(os.path.join(root, f"*.{e}")))
        self.files = sorted(self.files)

        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {root}")

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
        img_t = self.transform(img)
        return img_t, os.path.basename(p)


# ============================================================
# 2. 기본 DRAEM random anomaly 생성 함수 (CycleGAN 없음)
# ============================================================

def generate_random_anomaly(x, min_ratio=0.02, max_ratio=0.15, strength=1.0):
    """
    x: (B, 3, H, W), [-1, 1] 범위(normalize(mean=0.5, std=0.5) 기준)
    반환:
        x_anom: synthetic anomaly 이미지 (B, 3, H, W)
        mask  : anomaly 영역 (B, 1, H, W), {0,1}
    """
    B, C, H, W = x.shape
    device = x.device

    # 저해상도 노이즈 생성 후 업샘플링
    noise = torch.rand(B, 1, H // 4, W // 4, device=device)
    noise = nn.functional.interpolate(
        noise, size=(H, W),
        mode="bilinear",
        align_corners=False
    )

    mask = torch.zeros_like(noise)

    for b in range(B):
        flat = noise[b, 0].view(-1)
        numel = flat.numel()
        target_ratio = float(np.random.uniform(min_ratio, max_ratio))
        k = max(1, int(numel * target_ratio))
        topk_vals, _ = torch.topk(flat, k)
        thr = topk_vals[-1]

        mb = (noise[b, 0] >= thr).float().view(1, H, W)

        # dilation 비슷하게 주변까지 확장
        mb = nn.functional.max_pool2d(mb, kernel_size=3, stride=1, padding=1)
        mb = (mb > 0).float()

        mask[b] = mb

    # anomaly 텍스처 생성 ([-1,1] 범위 랜덤)
    anomaly_tex = (torch.rand_like(x) * 2.0 - 1.0) * strength

    # 정상 영역은 x, anomaly 영역은 anomaly_tex
    x_anom = x * (1.0 - mask) + anomaly_tex * mask
    return x_anom, mask


# ============================================================
# 3. 텐서를 저장 가능한 이미지로 변환
# ============================================================

def tensor_to_uint8(img_t):
    """
    img_t: (3, H, W), normalize(mean=0.5, std=0.5) 기준
    반환: (H, W, 3) uint8, [0,255]
    """
    img = img_t.detach().cpu()
    # [-1,1] -> [0,1]
    img = (img * 0.5 + 0.5).clamp(0.0, 1.0)
    img = img.permute(1, 2, 0).numpy()  # HWC
    img = (img * 255.0).round().astype(np.uint8)
    return img


def mask_to_uint8(mask_t):
    """
    mask_t: (1, H, W), 0 또는 1
    반환: (H, W) uint8, {0,255}
    """
    m = mask_t.detach().cpu()[0]  # (H, W)
    m = (m > 0.5).float().numpy()
    m = (m * 255.0).round().astype(np.uint8)
    return m


# ============================================================
# 4. 메인: 정상 이미지 -> random anomaly 생성 후 저장
# ============================================================

def export_random_anomalies(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(opt.out_dir, exist_ok=True)

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading normal images from: {opt.normal_dir}")

    dataset = NormalDataset(opt.normal_dir, img_size=opt.img_size)
    loader = DataLoader(dataset,
                        batch_size=opt.batch_size,
                        shuffle=False,
                        num_workers=opt.num_workers)

    print(f"[INFO] #normal images = {len(dataset)}")

    idx_global = 0

    for imgs, fnames in tqdm(loader, desc="Generating random anomalies"):
        imgs = imgs.to(device)

        # 기본 DRAEM random anomaly만 사용 (GAN 없음)
        x_anom, mask = generate_random_anomaly(
            imgs,
            min_ratio=opt.rand_min_ratio,
            max_ratio=opt.rand_max_ratio,
            strength=opt.rand_strength
        )

        B = imgs.size(0)
        for b in range(B):
            idx_global += 1
            base_name = os.path.splitext(fnames[b])[0]

            # 원본 / anomaly / mask 저장
            orig_np = tensor_to_uint8(imgs[b])
            anom_np = tensor_to_uint8(x_anom[b])
            mask_np = mask_to_uint8(mask[b])

            # 파일 이름
            orig_path = os.path.join(
                opt.out_dir, f"{base_name}_orig.png"
            )
            anom_path = os.path.join(
                opt.out_dir, f"{base_name}_anom.png"
            )
            mask_path = os.path.join(
                opt.out_dir, f"{base_name}_mask.png"
            )

            Image.fromarray(orig_np).save(orig_path)
            Image.fromarray(anom_np).save(anom_path)
            Image.fromarray(mask_np).save(mask_path)

    print(f"[INFO] Done. Saved synthetic anomalies to: {opt.out_dir}")


# ============================================================
# 5. Argparse
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export synthetic anomalies using basic DRAEM random anomaly (no CycleGAN)."
    )

    parser.add_argument(
        "--normal_dir",
        type=str,
        default="./train_data/train/normal",
        help="정상 이미지가 있는 폴더 경로"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./outputs_draem_synthetic",
        help="synthetic anomaly 이미지를 저장할 폴더"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Resize할 이미지 크기 (정사각형)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="배치 크기"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader num_workers"
    )

    # DRAEM random anomaly 파라미터
    parser.add_argument(
        "--rand_min_ratio",
        type=float,
        default=0.02,
        help="anomaly 영역 최소 비율"
    )
    parser.add_argument(
        "--rand_max_ratio",
        type=float,
        default=0.15,
        help="anomaly 영역 최대 비율"
    )
    parser.add_argument(
        "--rand_strength",
        type=float,
        default=1.0,
        help="anomaly 텍스처 강도 (기본 1.0)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()
    export_random_anomalies(opt)
