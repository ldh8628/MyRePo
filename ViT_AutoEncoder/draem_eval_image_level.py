import os
import argparse
from glob import glob

import numpy as np
from PIL import Image
import cv2

import torch
from torchvision import transforms
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

# draem_basic.py 에서 네트워크 정의를 가져온다고 가정
from draem_basic import ReconstructionNet, SegmentNet


def load_model(ckpt_dir, device):
    ckpts = sorted(glob(os.path.join(ckpt_dir, "draem_epoch_*.pth")))
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
    return recon_net, seg_net


def get_image_paths(root):
    exts = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")
    paths = []
    for e in exts:
        paths.extend(glob(os.path.join(root, f"*.{e}")))
    return sorted(paths)


def compute_anomaly_score(img_path, recon_net, seg_net, transform, device, img_size=256):
    img_pil = Image.open(img_path).convert("RGB")
    x = transform(img_pil).unsqueeze(0).to(device)  # (1,3,H,W)

    with torch.no_grad():
        # reconstruction
        x_recon = recon_net(x)
        seg_input = torch.cat([x, x_recon], dim=1)  # (1,6,H,W)
        mask_logit = seg_net(seg_input)
        mask_prob = torch.sigmoid(mask_logit)[0, 0].cpu().numpy()  # (H,W)

    # 방법 1: 전체 픽셀 중 최댓값
    score_max = float(mask_prob.max())

    # (선택) 방법 2: 상위 1% 평균
    flat = mask_prob.flatten()
    k = max(1, int(len(flat) * 0.01))
    topk_mean = float(np.mean(np.sort(flat)[-k:]))

    # 여기서는 가장 단순하게 score_max 를 사용
    return score_max, mask_prob


def eval_image_level(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    recon_net, seg_net = load_model(opt.ckpt_dir, device)

    transform = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])

    normal_paths = get_image_paths(opt.normal_dir)
    defect_paths = get_image_paths(opt.defect_dir)
    print(f"[INFO] #normal={len(normal_paths)}, #defect={len(defect_paths)}")

    y_true = []
    y_score = []

    # 정상 = 0
    for p in normal_paths:
        s, _ = compute_anomaly_score(p, recon_net, seg_net, transform, device, opt.img_size)
        y_true.append(0)
        y_score.append(s)

    # 불량 = 1
    for p in defect_paths:
        s, _ = compute_anomaly_score(p, recon_net, seg_net, transform, device, opt.img_size)
        y_true.append(1)
        y_score.append(s)

    y_true = np.array(y_true, dtype=np.int32)
    y_score = np.array(y_score, dtype=np.float32)

    # 1) AUROC
    auroc = roc_auc_score(y_true, y_score)
    print(f"\n[RESULT] Image-level AUROC: {auroc:.4f}")

    # 2) Threshold 선택 (예: Youden index 최대 or 단순 0.5)
    # 여기서는 score 분포 기준으로 중간값 사용 (간단 예시)
    thr = float(np.median(y_score))
    print(f"[INFO] Simple threshold (median of scores): {thr:.4f}")

    y_pred = (y_score >= thr).astype(np.int32)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("\n[RESULT] Confusion Matrix (rows:true, cols:pred) [0=normal,1=defect]:")
    print(cm)

    # Precision, recall, F1 등
    print("\n[RESULT] Classification report:")
    print(classification_report(y_true, y_pred, target_names=["normal", "defect"]))


def parse_args():
    parser = argparse.ArgumentParser(description="DRAEM image-level evaluation")

    parser.add_argument("--normal_dir", type=str, default="./data/normal",
                        help="directory of normal images")
    parser.add_argument("--defect_dir", type=str, default="./data/defect",
                        help="directory of defect images")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts_draem",
                        help="where draem_epoch_*.pth are stored")
    parser.add_argument("--img_size", type=int, default=256)

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()
    eval_image_level(opt)
