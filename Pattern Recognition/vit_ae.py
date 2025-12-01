import os
import random
import math
from glob import glob

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


# =========================
# 1. 설정
# =========================

class CFG:
    img_size = 256
    patch_size = 16
    in_chans = 3

    embed_dim = 384
    depth = 6
    num_heads = 6

    decoder_embed_dim = 192
    decoder_depth = 4
    decoder_num_heads = 6

    train_epochs = 50
    batch_size = 16
    lr = 1e-4
    weight_decay = 1e-5

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = "./data"  # normal, defect 폴더가 있는 경로
    save_dir = "./outputs_vit_ae"
    model_path = os.path.join(save_dir, "vit_ae.pth")


os.makedirs(CFG.save_dir, exist_ok=True)


# =========================
# 2. 유틸
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# =========================
# 3. Dataset 정의
# =========================

class MetalDefectDataset(Dataset):
    """
    root/
      normal/*.jpg
      defect/*.jpg
    """

    def __init__(self, root, img_paths, labels, img_size=256):
        self.root = root
        self.img_paths = img_paths
        self.labels = labels
        self.img_size = img_size

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # [0,1] 범위 그대로 사용 (금속 변색 민감도 확보)
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        label = self.labels[idx]  # 0: normal, 1: defect

        img = Image.open(path).convert("RGB")
        img_t = self.transform(img)

        return img_t, label, os.path.basename(path)


def build_datasets(data_root: str, img_size: int):
    normal_paths = sorted(glob(os.path.join(data_root, "normal", "*")))
    defect_paths = sorted(glob(os.path.join(data_root, "defect", "*")))

    labels_normal = [0] * len(normal_paths)
    labels_defect = [1] * len(defect_paths)

    all_paths = normal_paths + defect_paths
    all_labels = labels_normal + labels_defect

    # 전체 shuffle 후 train/val/test 분리
    paired = list(zip(all_paths, all_labels))
    random.shuffle(paired)
    all_paths, all_labels = zip(*paired)

    # 비율 예시: train 70%, val 15%, test 15%
    n_total = len(all_paths)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    n_test = n_total - n_train - n_val

    train_paths = all_paths[:n_train]
    train_labels = all_labels[:n_train]
    val_paths = all_paths[n_train:n_train + n_val]
    val_labels = all_labels[n_train:n_train + n_val]
    test_paths = all_paths[n_train + n_val:]
    test_labels = all_labels[n_train + n_val:]

    # 학습은 "정상 데이터만" 사용 (label==0)
    train_paths_normal = [p for p, l in zip(train_paths, train_labels) if l == 0]
    train_labels_normal = [0] * len(train_paths_normal)

    train_ds = MetalDefectDataset(data_root, train_paths_normal, train_labels_normal, img_size)
    val_ds = MetalDefectDataset(data_root, val_paths, val_labels, img_size)
    test_ds = MetalDefectDataset(data_root, test_paths, test_labels, img_size)

    return train_ds, val_ds, test_ds


# =========================
# 4. ViT AutoEncoder 모델
#   - MAE 스타일 간단 구현
# =========================

class PatchEmbed(nn.Module):
    """이미지를 patch로 나누고 flatten + linear projection"""
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim,
                                          num_heads=num_heads,
                                          dropout=drop,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        # x: [B, N, D]
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        return x


class ViTAutoEncoder(nn.Module):
    def __init__(self,
                 img_size=256,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=384,
                 depth=6,
                 num_heads=6,
                 decoder_embed_dim=192,
                 decoder_depth=4,
                 decoder_num_heads=6):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # ---------- Encoder ----------
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])
        self.enc_norm = nn.LayerNorm(embed_dim)

        # ---------- Decoder ----------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # patch당 픽셀 수: 3 * patch_size * patch_size
        self.head = nn.Linear(decoder_embed_dim,
                              in_chans * patch_size * patch_size)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        # Linear / LayerNorm 기본 초기화는 PyTorch default 사용

    def patchify(self, imgs):
        """
        imgs: [B, C, H, W]
        return: [B, N, patch_size*patch_size*C]
        """
        B, C, H, W = imgs.shape
        p = self.patch_size
        assert H == W == self.img_size

        x = imgs.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # [B, H/p, W/p, p, p, C]
        patches = x.reshape(B, -1, p * p * C)
        return patches

    def unpatchify(self, patches):
        """
        patches: [B, N, patch_size*patch_size*C]
        return: [B, C, H, W]
        """
        B, N, D = patches.shape
        p = self.patch_size
        C = CFG.in_chans
        h = w = int(math.sqrt(N))
        x = patches.reshape(B, h, w, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4)  # [B, C, h, p, w, p]
        imgs = x.reshape(B, C, h * p, w * p)
        return imgs

    def forward(self, imgs):
        # Encoder
        x = self.patch_embed(imgs)  # [B, N, D]
        x = x + self.pos_embed

        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.enc_norm(x)

        # Decoder
        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.head(x)  # [B, N, patch_size*patch_size*C]
        recon_imgs = self.unpatchify(x)

        return recon_imgs


# =========================
# 5. 학습 루프
# =========================

def train_model(model, train_loader, val_loader):
    model.to(CFG.device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CFG.lr,
                                  weight_decay=CFG.weight_decay)

    best_val = float("inf")

    for epoch in range(1, CFG.train_epochs + 1):
        model.train()
        train_loss = 0.0

        for imgs, _, _ in tqdm(train_loader, desc=f"Train {epoch}", leave=False):
            imgs = imgs.to(CFG.device)
            optimizer.zero_grad()
            recon = model(imgs)
            loss = F.mse_loss(recon, imgs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # ---- validation (정상/불량 모두 포함, reconstruction error만 확인) ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, _, _ in val_loader:
                imgs = imgs.to(CFG.device)
                recon = model(imgs)
                loss = F.mse_loss(recon, imgs)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"[Epoch {epoch}] train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), CFG.model_path)
            print(f"  >> best model updated (val_loss={best_val:.6f})")

    print("Training finished.")


# =========================
# 6. 이상 점수 계산 + 불량/정상 판정
# =========================

def compute_recon_error_map(img, recon):
    """
    img, recon: [1, C, H, W] (tensor)
    return:
      error_map: [H, W] numpy (0~1 정규화)
      score: float (전역 점수)
    """
    with torch.no_grad():
        diff = (img - recon).pow(2).mean(dim=1, keepdim=True)  # [1,1,H,W]
        error_map = diff.squeeze().cpu().numpy()

    # 전역 점수 = 평균 오차
    score = float(error_map.mean())

    # 0~1 정규화
    em_min, em_max = error_map.min(), error_map.max()
    if em_max > em_min:
        error_map_norm = (error_map - em_min) / (em_max - em_min)
    else:
        error_map_norm = np.zeros_like(error_map)

    return error_map_norm, score


def evaluate_threshold(model, val_loader):
    """
    val set에서 normal / defect reconstruction error 분포를 보고
    간단히 threshold를 정한다.
    """
    model.to(CFG.device)
    model.eval()

    normal_scores = []
    defect_scores = []

    with torch.no_grad():
        for imgs, labels, _ in tqdm(val_loader, desc="Calc threshold", leave=False):
            imgs = imgs.to(CFG.device)
            recon = model(imgs)
            diff = (imgs - recon).pow(2).mean(dim=[1, 2, 3])  # batch-wise score

            for s, y in zip(diff.cpu().numpy(), labels.numpy()):
                if y == 0:
                    normal_scores.append(float(s))
                else:
                    defect_scores.append(float(s))

    normal_scores = np.array(normal_scores)
    defect_scores = np.array(defect_scores)

    print(f"Normal scores: mean={normal_scores.mean():.6f}, std={normal_scores.std():.6f}")
    print(f"Defect scores: mean={defect_scores.mean():.6f}, std={defect_scores.std():.6f}")

    # 간단한 기준: normal 평균 + 3*std
    threshold = normal_scores.mean() + 3 * normal_scores.std()
    print(f"Initial threshold = {threshold:.6f}")

    return threshold, normal_scores, defect_scores


def save_heatmap_overlay(img_np, error_map, save_path, alpha=0.5):
    """
    img_np: [H,W,3] 0~255
    error_map: [H,W] 0~1
    """
    cmap = plt.get_cmap("jet")
    heatmap = cmap(error_map)[:, :, :3]  # RG B
    heatmap = (heatmap * 255).astype(np.uint8)

    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def test_and_visualize(model, test_loader, threshold):
    model.to(CFG.device)
    model.eval()

    os.makedirs(os.path.join(CFG.save_dir, "vis"), exist_ok=True)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for imgs, labels, names in tqdm(test_loader, desc="Test", leave=False):
            imgs = imgs.to(CFG.device)
            recon = model(imgs)

            for i in range(imgs.size(0)):
                img = imgs[i:i+1]
                rec = recon[i:i+1]
                label = int(labels[i].item())
                name = names[i]

                error_map, score = compute_recon_error_map(img, rec)

                pred = 1 if score > threshold else 0  # 1: defect
                y_true.append(label)
                y_pred.append(pred)

                # 시각화 저장 (불량 또는 점수가 threshold 근처인 이미지 위주로)
                if label == 1 or abs(score - threshold) / threshold < 0.2:
                    img_np = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    save_path = os.path.join(CFG.save_dir, "vis",
                                             f"{name}_label{label}_score{score:.4f}.png")
                    save_heatmap_overlay(img_np, error_map, save_path)

    # 간단한 성능 지표
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = (y_true == y_pred).mean()
    from sklearn.metrics import precision_score, recall_score, f1_score
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Test ACC={acc:.4f}, Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}")
    print(f"Heatmap overlay images are saved in: {os.path.join(CFG.save_dir, 'vis')}")


# =========================
# 7. main
# =========================

def main():
    print(f"Using device: {CFG.device}")

    # --- dataset ---
    train_ds, val_ds, test_ds = build_datasets(CFG.data_root, CFG.img_size)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CFG.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # --- model ---
    model = ViTAutoEncoder(
        img_size=CFG.img_size,
        patch_size=CFG.patch_size,
        in_chans=CFG.in_chans,
        embed_dim=CFG.embed_dim,
        depth=CFG.depth,
        num_heads=CFG.num_heads,
        decoder_embed_dim=CFG.decoder_embed_dim,
        decoder_depth=CFG.decoder_depth,
        decoder_num_heads=CFG.decoder_num_heads
    )

    # --- training ---
    if not os.path.exists(CFG.model_path):
        print("Start training ViT AutoEncoder...")
        train_model(model, train_loader, val_loader)
    else:
        print(f"Load pretrained model: {CFG.model_path}")
        model.load_state_dict(torch.load(CFG.model_path, map_location=CFG.device))

    # --- threshold estimation ---
    threshold, _, _ = evaluate_threshold(model, val_loader)

    # --- test + visualization ---
    test_and_visualize(model, test_loader, threshold)


if __name__ == "__main__":
    main()
