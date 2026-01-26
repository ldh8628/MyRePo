import os
import random
from glob import glob

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


# =========================
# 1. 설정
# =========================

class CFG:
    img_size = 256

    batch_size = 16

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = "./data"       # normal, defect 폴더가 있는 경로
    save_dir = "./outputs_patchcore"

    # PatchCore 관련
    coreset_rate = 0.01        # 정상 패치 중 몇 %를 메모리 뱅크로 쓸지 (0.01 = 1%)
    feature_map_size = 32      # 최종 feature map 크기 (HxW)

    # DataLoader 설정 (Windows용 자원 절약)
    dataloader_workers = 0     # 0으로 두면 공유 메모리 에러 방지
    pin_memory = False         # shared memory 사용 최소화


os.makedirs(CFG.save_dir, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# =========================
# 2. Dataset
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
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        label = self.labels[idx]  # 0 normal, 1 defect
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

    paired = list(zip(all_paths, all_labels))
    random.shuffle(paired)
    all_paths, all_labels = zip(*paired)

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

    # PatchCore 학습에는 정상 train 데이터만 사용
    train_paths_normal = [p for p, l in zip(train_paths, train_labels) if l == 0]
    train_labels_normal = [0] * len(train_paths_normal)

    train_ds = MetalDefectDataset(data_root, train_paths_normal, train_labels_normal, img_size)
    val_ds = MetalDefectDataset(data_root, val_paths, val_labels, img_size)
    test_ds = MetalDefectDataset(data_root, test_paths, test_labels, img_size)

    return train_ds, val_ds, test_ds


# =========================
# 3. Backbone (ResNet50) + Feature Extractor
# =========================

class ResNetEmbedder(nn.Module):
    """
    ResNet50에서 layer2, layer3 feature를 추출해서
    PatchCore용 patch feature로 변환
    """
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(pretrained=True)

        # ResNet 구조 분해
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        # layer4는 사용하지 않음

    def forward(self, x):
        # x: [B,3,H,W]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   # [B,64,H/4,W/4]

        x1 = self.layer1(x)   # [B,256,...]
        x2 = self.layer2(x1)  # [B,512,H/8,W/8]
        x3 = self.layer3(x2)  # [B,1024,H/16,W/16]

        # feature size 정규화 (예: 32x32로 맞추기)
        target_size = CFG.feature_map_size

        f2 = F.interpolate(x2, size=(target_size, target_size),
                           mode="bilinear", align_corners=False)
        f3 = F.interpolate(x3, size=(target_size, target_size),
                           mode="bilinear", align_corners=False)

        feat = torch.cat([f2, f3], dim=1)  # [B, C2+C3, H, W]
        return feat


def extract_patch_features(dataloader, model):
    """
    dataloader의 모든 이미지에 대해 patch feature 추출
    return:
      all_features: list of (N_i, C) numpy
      all_labels: list of int (이미지 라벨)
      shapes: list of (H, W)  # feature map spatial 크기
      names: list of 파일명
    """
    model.to(CFG.device)
    model.eval()

    all_features = []
    all_labels = []
    shapes = []
    names = []

    with torch.no_grad():
        for imgs, labels, fnames in tqdm(dataloader, desc="Extract features", leave=False):
            imgs = imgs.to(CFG.device)
            feats = model(imgs)              # [B,C,H,W]
            B, C, H, W = feats.shape

            feats = F.normalize(feats, p=2, dim=1)  # 채널 기준 L2 normalize
            feats = feats.permute(0, 2, 3, 1)       # [B,H,W,C]
            feats = feats.reshape(B, H * W, C)      # [B,N,C]

            feats_np = feats.cpu().numpy()

            for i in range(B):
                all_features.append(feats_np[i])    # [N,C]
                all_labels.append(int(labels[i]))
                shapes.append((H, W))
                names.append(fnames[i])

    return all_features, all_labels, shapes, names


# =========================
# 4. 메모리 뱅크(coreset) 구성 (메모리 절약 버전)
# =========================

def build_feature_bank(train_features,
                       max_patches_per_image=300,
                       use_fp16=True):
    """
    train_features: list of (N_i, C) numpy (train normal만)

    - 이미지당 patch 수를 max_patches_per_image로 제한
    - 필요 시 float16으로 downcast
    → np.concatenate 시 메모리 폭발 방지
    """
    feats_list = []

    for f in train_features:   # f: [N_i, C]
        N, C = f.shape

        # 이미지당 patch 수 제한
        if N > max_patches_per_image:
            idx = np.random.choice(N, max_patches_per_image, replace=False)
            f = f[idx]

        if use_fp16:
            f = f.astype(np.float16)

        feats_list.append(f)

    feats = np.concatenate(feats_list, axis=0)  # [N_total_reduced, C]
    n_total = feats.shape[0]
    n_coreset = max(1, int(n_total * CFG.coreset_rate))

    print(f"Total (reduced) normal patches: {n_total}")
    print(f"Coreset rate: {CFG.coreset_rate} -> {n_coreset} patches used")

    idx = np.random.choice(n_total, n_coreset, replace=False)
    coreset = feats[idx]

    # Nearest Neighbor 모델 구성 (CPU)
    nn_model = NearestNeighbors(n_neighbors=1, algorithm="auto")
    # NearestNeighbors는 float32를 사용하는 것이 안전하므로 다시 float32로 변환
    nn_model.fit(coreset.astype(np.float32))

    return coreset, nn_model


# =========================
# 5. Heatmap 및 시각화
# =========================

def save_heatmap_overlay(img_np, anomaly_map, save_path, alpha=0.6):
    """
    img_np: [H,W,3] uint8 (RGB)
    anomaly_map: [H,W] float (0~1)
    """
    cmap = plt.get_cmap("jet")
    heatmap = cmap(anomaly_map)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


# =========================
# 6. threshold 계산 및 테스트
# =========================

def calc_scores(all_features, all_labels, shapes, nn_model):
    """
    PatchCore anomaly score 계산
    - patch distance 중 max 값을 이미지 score로 사용
    - patch distance map은 후에 heatmap으로 사용
    """
    img_scores = []
    patch_maps = []

    for feat, label, (H, W) in zip(all_features, all_labels, shapes):
        # feat: [N,C]
        dist, _ = nn_model.kneighbors(feat, n_neighbors=1, return_distance=True)
        dist = dist.reshape(H, W)

        score = float(dist.max())
        img_scores.append(score)
        patch_maps.append(dist)

    return np.array(img_scores), patch_maps


def estimate_threshold(val_scores, val_labels):
    """
    정상/불량 val score 분포를 기반으로 threshold 추정
    간단하게: 정상 평균 + 3*표준편차
    """
    val_labels = np.array(val_labels)
    normal_scores = val_scores[val_labels == 0]
    defect_scores = val_scores[val_labels == 1]

    print(f"Val normal scores: mean={normal_scores.mean():.6f}, std={normal_scores.std():.6f}")
    print(f"Val defect  scores: mean={defect_scores.mean():.6f}, std={defect_scores.std():.6f}")

    thr = normal_scores.mean() + 3 * normal_scores.std()
    print(f"Initial threshold: {thr:.6f}")
    return thr


def evaluate_predictions(scores, labels, threshold):
    """
    scores: anomaly score (높을수록 불량)
    labels: 0=normal, 1=defect
    """
    labels = np.array(labels)
    preds = (scores > threshold).astype(np.int32)  # 1: defect

    acc = (preds == labels).mean()
    p = precision_score(labels, preds, zero_division=0)
    r = recall_score(labels, preds, zero_division=0)  # defect(1)에 대한 recall
    f1 = f1_score(labels, preds, zero_division=0)
    
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # normal / defect 별 recall 계산
    normal_recall = tn / (tn + fp + 1e-8)
    defect_recall = tp / (tp + fn + 1e-8)

    return acc, p, r, f1, preds, cm, (tn, fp, fn, tp), normal_recall, defect_recall


def test_and_visualize(test_loader, backbone, nn_model, threshold):
    backbone.to(CFG.device)
    backbone.eval()

    os.makedirs(os.path.join(CFG.save_dir, "vis"), exist_ok=True)

    # feature 추출
    test_feats, test_labels, test_shapes, test_names = extract_patch_features(test_loader, backbone)

    test_scores, patch_maps = calc_scores(test_feats,
                                          np.array(test_labels),
                                          test_shapes,
                                          nn_model)

    acc, p, r, f1, preds, cm, (tn, fp, fn, tp), normal_recall, defect_recall = evaluate_predictions(
        test_scores, np.array(test_labels), threshold
    )

    labels_np = np.array(test_labels)
    auroc = roc_auc_score(labels_np, test_scores)

    print(f"[Test] AUROC={auroc:.4f}, ACC={acc:.4f}, Precision={p:.4f}, Recall(defect)={r:.4f}, F1={f1:.4f}")
    print(f"TN:{tn}, FP:{fp}, FN:{fn}, TP:{tp}")
    print(f"normal_recall={normal_recall:.4f}, defect_recall={defect_recall:.4f}")

    # heatmap 시각화 (불량 또는 경계 근처 위주)
    for feat, label, score, amap, (H, W), name, (img_t, _, _) in zip(
            test_feats, test_labels, test_scores, patch_maps, test_shapes,
            test_names, test_loader.dataset):

        if label == 1 or abs(score - threshold) / (threshold + 1e-8) < 0.2:
            # 원본 이미지 복원
            img_tensor, _, _ = img_t, label, name
            img_np = (img_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            # anomaly map을 원본 크기로 resize
            amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
            amap_resized = cv2.resize(amap_norm,
                                      (img_np.shape[1], img_np.shape[0]),
                                      interpolation=cv2.INTER_CUBIC)

            save_path = os.path.join(
                CFG.save_dir, "vis",
                f"{name}_label{label}_score{score:.4f}.png"
            )
            save_heatmap_overlay(img_np, amap_resized, save_path)


# =========================
# 7. main
# =========================

def main():
    print(f"Using device: {CFG.device}")

    train_ds, val_ds, test_ds = build_datasets(CFG.data_root, CFG.img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.dataloader_workers,
        pin_memory=CFG.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.dataloader_workers,
        pin_memory=CFG.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.dataloader_workers,
        pin_memory=CFG.pin_memory,
    )

    backbone = ResNetEmbedder()

    # ---- 1) train normal에서 feature bank 구축 ----
    train_feats, train_labels, train_shapes, _ = extract_patch_features(train_loader, backbone)

    # train_labels는 전부 0(normal)이므로 별도 사용 X
    feature_bank, nn_model = build_feature_bank(train_feats)

    # ---- 2) val에서 threshold 산출 ----
    val_feats, val_labels, val_shapes, _ = extract_patch_features(val_loader, backbone)
    val_scores, _ = calc_scores(val_feats,
                                np.array(val_labels),
                                val_shapes,
                                nn_model)
    threshold = estimate_threshold(val_scores, np.array(val_labels))

    # ---- 3) test에서 성능 평가 + heatmap 저장 ----
    test_and_visualize(test_loader, backbone, nn_model, threshold)


if __name__ == "__main__":
    main()
