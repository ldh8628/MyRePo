import os
from glob import glob
import argparse

import numpy as np
import cv2
from PIL import Image

import torch
from torchvision import transforms
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt

# draem_hybrid_gan.py 에서 네트워크 가져오기
from draem_hybrid_gan import ReconstructionNet, SegmentNet


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


def get_image_paths(root):
    exts = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")
    paths = []
    for e in exts:
        paths.extend(glob(os.path.join(root, f"*.{e}")))
    return sorted(paths)


def compute_score(
    img_path,
    recon_net,
    seg_net,
    transform,
    device,
    img_size,
    save_root=None,
    label_name=None,
):
    """
    anomaly score = max(mask_prob).
    save_root, label_name 이 주어지면 heatmap/overlay PNG 저장.
    """
    img_pil = Image.open(img_path).convert("RGB")
    x = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        x_recon = recon_net(x)
        seg_input = torch.cat([x, x_recon], dim=1)
        mask_logit = seg_net(seg_input)
        mask_prob = torch.sigmoid(mask_logit)[0, 0].cpu().numpy()

    score = float(mask_prob.max())

    if save_root is not None and label_name is not None:
        heat = (mask_prob * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

        img_np = np.array(img_pil.resize((img_size, img_size)))
        overlay = cv2.addWeighted(img_np, 0.6, heat_color, 0.4, 0)

        out_dir = os.path.join(save_root, label_name)
        os.makedirs(out_dir, exist_ok=True)

        fname = os.path.basename(img_path)
        base, _ = os.path.splitext(fname)
        out_heat = os.path.join(out_dir, base + "_heat.png")
        out_overlay = os.path.join(out_dir, base + "_overlay.png")

        cv2.imwrite(out_heat, cv2.cvtColor(heat_color, cv2.COLOR_RGB2BGR))
        cv2.imwrite(out_overlay, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return score


def main(opt):
    os.makedirs(opt.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    recon_net, seg_net, ckpt_path = load_latest_ckpt(opt.ckpt_dir, device)

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
    filenames = []

    # score 계산 + (옵션) heatmap 저장
    for p in normal_paths:
        s = compute_score(
            p,
            recon_net,
            seg_net,
            transform,
            device,
            opt.img_size,
            save_root=opt.save_images_dir if opt.save_images else None,
            label_name="normal" if opt.save_images else None,
        )
        y_true.append(0)
        y_score.append(s)
        filenames.append(os.path.basename(p))

    for p in defect_paths:
        s = compute_score(
            p,
            recon_net,
            seg_net,
            transform,
            device,
            opt.img_size,
            save_root=opt.save_images_dir if opt.save_images else None,
            label_name="defect" if opt.save_images else None,
        )
        y_true.append(1)
        y_score.append(s)
        filenames.append(os.path.basename(p))

    y_true = np.array(y_true, dtype=np.int32)
    y_score = np.array(y_score, dtype=np.float32)

    # ---------------------------
    # 1) AUROC
    # ---------------------------
    auroc = roc_auc_score(y_true, y_score)
    print(f"\n[RESULT] Image-level AUROC: {auroc:.6f}")

    # ---------------------------
    # 2) ROC curve + Youden index 로 best threshold
    # ---------------------------
    fpr, tpr, thr = roc_curve(y_true, y_score)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_thr = float(thr[best_idx])
    print(f"[INFO] Best threshold (Youden index): {best_thr:.6f}")
    print(f"[INFO] TPR={tpr[best_idx]:.4f}, FPR={fpr[best_idx]:.4f}")

    # ---------------------------
    # 3) Confusion matrix at best threshold
    # ---------------------------
    y_pred_best = (y_score >= best_thr).astype(np.int32)
    cm_best = confusion_matrix(y_true, y_pred_best, labels=[0, 1])
    report_best = classification_report(
        y_true, y_pred_best,
        target_names=["normal", "defect"],
        digits=4,
    )

    print("\n[RESULT] Confusion Matrix @ best_thr (rows:true, cols:pred) [0=normal,1=defect]:")
    print(cm_best)
    print("\n[RESULT] Classification report @ best_thr:")
    print(report_best)

    # ---------------------------
    # 4) (참고) median threshold 도 같이 계산
    # ---------------------------
    median_thr = float(np.median(y_score))
    y_pred_med = (y_score >= median_thr).astype(np.int32)
    cm_med = confusion_matrix(y_true, y_pred_med, labels=[0, 1])
    report_med = classification_report(
        y_true, y_pred_med,
        target_names=["normal", "defect"],
        digits=4,
    )

    print(f"\n[INFO] Median threshold: {median_thr:.6f}")
    print("\n[RESULT] Confusion Matrix @ median_thr:")
    print(cm_med)
    print("\n[RESULT] Classification report @ median_thr:")
    print(report_med)

    # ---------------------------
    # 5) metrics 파일 저장
    # ---------------------------
    metrics_path = os.path.join(opt.out_dir, "metrics_with_threshold.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Image-level AUROC: {auroc:.6f}\n\n")

        f.write(f"[Best threshold (Youden)] {best_thr:.6f}\n")
        f.write(f"TPR={tpr[best_idx]:.6f}, FPR={fpr[best_idx]:.6f}\n")
        f.write("Confusion Matrix (rows:true, cols:pred) [0=normal,1=defect]:\n")
        f.write(str(cm_best) + "\n\n")
        f.write("Classification report:\n")
        f.write(report_best + "\n\n")

        f.write(f"[Median threshold] {median_thr:.6f}\n")
        f.write("Confusion Matrix (rows:true, cols:pred) [0=normal,1=defect]:\n")
        f.write(str(cm_med) + "\n\n")
        f.write("Classification report:\n")
        f.write(report_med + "\n")
    print(f"[INFO] Saved metrics to {metrics_path}")

    # ---------------------------
    # 6) score 분포 히스토그램 / ROC curve 저장
    # ---------------------------
    # score 분포
    scores_normal = y_score[y_true == 0]
    scores_defect = y_score[y_true == 1]

    plt.figure()
    plt.hist(scores_normal, bins=50, alpha=0.7, label="normal")
    plt.hist(scores_defect, bins=50, alpha=0.7, label="defect")
    plt.axvline(best_thr, color="k", linestyle="--", label=f"best_thr={best_thr:.4f}")
    plt.xlabel("anomaly score")
    plt.ylabel("count")
    plt.title("Score distribution (normal vs defect)")
    plt.legend()
    plt.tight_layout()
    hist_path = os.path.join(opt.out_dir, "score_histogram.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"[INFO] Saved score histogram to {hist_path}")

    # ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUROC={auroc:.4f})")
    plt.scatter(fpr[best_idx], tpr[best_idx], color="red", label="best_thr")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend()
    plt.tight_layout()
    roc_path = os.path.join(opt.out_dir, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"[INFO] Saved ROC curve to {roc_path}")

    # ---------------------------
    # 7) scores.csv 저장 (파일별 score/label/pred_best)
    # ---------------------------
    csv_path = os.path.join(opt.out_dir, "scores_with_best_threshold.csv")
    with open(csv_path, "w") as f:
        f.write("filename,label,score,pred_best\n")
        for fn, y, s, yp in zip(filenames, y_true, y_score, y_pred_best):
            f.write(f"{fn},{y},{s:.6f},{int(yp)}\n")
    print(f"[INFO] Saved scores to {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="DRAEM hybrid: threshold search + score distribution plots"
    )

    parser.add_argument("--normal_dir", type=str, default="./data/normal")
    parser.add_argument("--defect_dir", type=str, default="./data/defect")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts_draem_hybrid")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--out_dir", type=str, default="./eval_threshold")

    # heatmap/overlay 이미지도 같이 저장할지 여부
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--save_images_dir", type=str, default="./eval_threshold_images")

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
