import os
from glob import glob
import argparse

import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from draem_hybrid_split import ReconstructionNet, SegmentNet, get_image_paths


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


def compute_score(img_path, recon_net, seg_net, transform, device):
    img_pil = Image.open(img_path).convert("RGB")
    x = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        x_recon = recon_net(x)
        seg_input = torch.cat([x, x_recon], dim=1)
        mask_logit = seg_net(seg_input)
        mask_prob = torch.sigmoid(mask_logit)[0, 0].cpu().numpy()

    score = float(mask_prob.max())
    return score


def eval_split_once(data_root, split_name, recon_net, seg_net, img_size, device):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])

    normal_dir = os.path.join(data_root, split_name, "normal")
    defect_dir = os.path.join(data_root, split_name, "defect")

    normal_paths = get_image_paths(normal_dir)
    defect_paths = get_image_paths(defect_dir)

    print(f"[INFO] [{split_name}] #normal={len(normal_paths)}, #defect={len(defect_paths)}")

    y_true = []
    y_score = []

    for p in normal_paths:
        s = compute_score(p, recon_net, seg_net, transform, device)
        y_true.append(0)
        y_score.append(s)

    for p in defect_paths:
        s = compute_score(p, recon_net, seg_net, transform, device)
        y_true.append(1)
        y_score.append(s)

    y_true = np.array(y_true, dtype=np.int32)
    y_score = np.array(y_score, dtype=np.float32)
    return y_true, y_score


def main(opt):
    os.makedirs(opt.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    recon_net, seg_net, ckpt_path = load_latest_ckpt(opt.ckpt_dir, device)

    # 1) validation에서 AUROC + best threshold
    y_true_val, y_score_val = eval_split_once(
        opt.data_root, "validation", recon_net, seg_net, opt.img_size, device
    )

    auroc_val = roc_auc_score(y_true_val, y_score_val)
    fpr, tpr, thr = roc_curve(y_true_val, y_score_val)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_thr = float(thr[best_idx])

    print(f"\n[VAL] AUROC={auroc_val:.6f}")
    print(f"[VAL] Best threshold (Youden)={best_thr:.6f}, TPR={tpr[best_idx]:.4f}, FPR={fpr[best_idx]:.4f}")

    y_pred_val = (y_score_val >= best_thr).astype(np.int32)
    cm_val = confusion_matrix(y_true_val, y_pred_val, labels=[0, 1])
    report_val = classification_report(
        y_true_val, y_pred_val,
        target_names=["normal", "defect"],
        digits=4,
    )

    print("\n[VAL] Confusion Matrix:")
    print(cm_val)
    print("\n[VAL] Classification report:")
    print(report_val)

    # 2) test에서는 best_thr를 고정하여 사용
    y_true_test, y_score_test = eval_split_once(
        opt.data_root, "test", recon_net, seg_net, opt.img_size, device
    )

    auroc_test = roc_auc_score(y_true_test, y_score_test)
    y_pred_test = (y_score_test >= best_thr).astype(np.int32)
    cm_test = confusion_matrix(y_true_test, y_pred_test, labels=[0, 1])
    report_test = classification_report(
        y_true_test, y_pred_test,
        target_names=["normal", "defect"],
        digits=4,
    )

    print(f"\n[TEST] AUROC={auroc_test:.6f}  (threshold from VAL)")
    print("\n[TEST] Confusion Matrix (VAL threshold):")
    print(cm_test)
    print("\n[TEST] Classification report (VAL threshold):")
    print(report_test)

    # 3) 결과 파일 저장
    metrics_path = os.path.join(opt.out_dir, "metrics_val_test.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Checkpoint: {ckpt_path}\n")

        f.write(f"\n[VAL] AUROC={auroc_val:.6f}\n")
        f.write(f"[VAL] Best threshold (Youden)={best_thr:.6f}\n")
        f.write("Confusion Matrix (rows:true, cols:pred) [0=normal,1=defect]:\n")
        f.write(str(cm_val) + "\n\n")
        f.write("Classification report:\n")
        f.write(report_val + "\n")

        f.write(f"\n[TEST] AUROC={auroc_test:.6f}  (threshold from VAL)\n")
        f.write("Confusion Matrix (rows:true, cols:pred) [0=normal,1=defect]:\n")
        f.write(str(cm_test) + "\n\n")
        f.write("Classification report:\n")
        f.write(report_test + "\n")

    print(f"[INFO] Saved metrics to {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DRAEM with val-chosen threshold on test split."
    )
    parser.add_argument("--data_root", type=str, default="./train_data")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts_draem_hybrid_split")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--out_dir", type=str, default="./eval_val_test")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
