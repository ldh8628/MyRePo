#!/usr/bin/env bash
set -euo pipefail
root="${1:-}"
cad="${2:-}"
camera="${3:-}"
templates="${4:-}"
[[ -n "$root" && -d "$root/Instance_Segmentation_Model" && -d "$root/Pose_Estimation_Model" ]] || { echo "ERROR: argument 1 must be a SAM-6D checkout" >&2; exit 1; }
[[ -n "$cad" && -f "$cad" ]] || { echo "ERROR: CAD .ply missing (millimetre units required)" >&2; exit 1; }
[[ -n "$camera" && -f "$camera" ]] || { echo "ERROR: camera intrinsics JSON missing" >&2; exit 1; }
[[ -n "$templates" && -d "$templates" ]] || { echo "ERROR: rendered template directory missing" >&2; exit 1; }
for pattern in 'rgb_*.png' 'mask_*.png' 'xyz_*.npy'; do
  find "$templates" -type f -name "$pattern" -print -quit | grep -q . || { echo "ERROR: template lacks $pattern" >&2; exit 1; }
done
required_weights=(
  "$root/Pose_Estimation_Model/checkpoints/sam-6d-pem-base.pth"
  "$root/Instance_Segmentation_Model/checkpoints/FastSAM/FastSAM-s.pt"
  "$root/Instance_Segmentation_Model/checkpoints/dinov2/dinov2_vits14_pretrain.pth"
)
for weight in "${required_weights[@]}"; do
  [[ -s "$weight" ]] || { echo "ERROR: required checkpoint missing: $weight; use upstream prepare.sh" >&2; exit 1; }
done
echo "SAM-6D assets OK"
