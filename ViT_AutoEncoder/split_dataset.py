import os
import shutil
import random
import argparse
from glob import glob


def collect_images(root_dir):
    """
    root_dir 안의 이미지 파일 경로 리스트를 반환.
    """
    exts = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")
    files = []
    for e in exts:
        files.extend(glob(os.path.join(root_dir, f"*.{e}")))
    return sorted(files)


def split_list(lst, train_ratio, val_ratio, test_ratio, seed=42):
    """
    lst 를 train/val/test 로 랜덤 split.
    train_ratio + val_ratio + test_ratio == 1.0 이어야 함.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train_ratio + val_ratio + test_ratio must be 1.0"

    random.seed(seed)
    lst = lst[:]  # copy
    random.shuffle(lst)

    n = len(lst)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val  # 나머지 전부 test

    train_list = lst[:n_train]
    val_list = lst[n_train:n_train + n_val]
    test_list = lst[n_train + n_val:]

    return train_list, val_list, test_list


def safe_copy(src_list, dst_dir):
    """
    src_list 의 파일들을 dst_dir 로 copy2.
    dst_dir 이 없으면 생성.
    """
    os.makedirs(dst_dir, exist_ok=True)
    for src in src_list:
        fname = os.path.basename(src)
        dst = os.path.join(dst_dir, fname)
        shutil.copy2(src, dst)


def main(opt):
    # 1. 원본 폴더에서 이미지 수집
    normal_files = collect_images(opt.normal_dir)
    defect_files = collect_images(opt.defect_dir)

    print(f"[INFO] #normal images = {len(normal_files)} from {opt.normal_dir}")
    print(f"[INFO] #defect images = {len(defect_files)} from {opt.defect_dir}")

    if len(normal_files) == 0:
        raise RuntimeError("No normal images found. Check normal_dir.")
    if len(defect_files) == 0:
        print("[WARN] No defect images found. Defect split will be empty.")

    # 2-1. normal 은 train/val/test 모두 사용 (예: 0.7/0.15/0.15)
    normal_train, normal_val, normal_test = split_list(
        normal_files, opt.normal_train_ratio, opt.normal_val_ratio,
        opt.normal_test_ratio, seed=opt.seed
    )

    # 2-2. defect 는 train=0, val/test 만 사용 (예: 0.0/0.3/0.7)
    defect_train, defect_val, defect_test = split_list(
        defect_files, 0.0, opt.defect_val_ratio, opt.defect_test_ratio, seed=opt.seed
    )
    # defect_train 은 사용하지 않음

    print(f"[INFO] Normal split: train={len(normal_train)}, "
          f"val={len(normal_val)}, test={len(normal_test)}")
    print(f"[INFO] Defect split: train={len(defect_train)}(unused), "
          f"val={len(defect_val)}, test={len(defect_test)}")

    # 3. 복사 대상 폴더 구조
    train_root = opt.out_root

    train_normal_dir = os.path.join(train_root, "train", "normal")
    train_defect_dir = os.path.join(train_root, "train", "defect")
    val_normal_dir = os.path.join(train_root, "validation", "normal")
    val_defect_dir = os.path.join(train_root, "validation", "defect")
    test_normal_dir = os.path.join(train_root, "test", "normal")
    test_defect_dir = os.path.join(train_root, "test", "defect")

    # 기존 내용이 남는 게 싫으면 train_root 전체를 지우고 시작하는 것도 가능
    # (주의: 이미 있는 데이터를 모두 삭제함)
    # if os.path.exists(train_root):
    #     shutil.rmtree(train_root)

    print(f"[INFO] Copying normal images...")
    safe_copy(normal_train, train_normal_dir)
    safe_copy(normal_val, val_normal_dir)
    safe_copy(normal_test, test_normal_dir)

    print(f"[INFO] Copying defect images (no train defect)...")
    # train_defect_dir 에는 복사하지 않음
    safe_copy(defect_val, val_defect_dir)
    safe_copy(defect_test, test_defect_dir)

    print("[INFO] Done. Split dataset created at:", train_root)
    print("  - train/normal :", train_normal_dir)
    print("  - train/defect : (not used in this script)")
    print("  - validation/normal :", val_normal_dir)
    print("  - validation/defect :", val_defect_dir)
    print("  - test/normal :", test_normal_dir)
    print("  - test/defect :", test_defect_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Random split copy from data/normal & data/defect "
                     "to train_data/train|validation|test folders "
                     "(defect train=0, val/test only).")
    )

    parser.add_argument("--normal_dir", type=str, default="./data/normal",
                        help="Path to original normal images")
    parser.add_argument("--defect_dir", type=str, default="./data/defect",
                        help="Path to original defect images")

    parser.add_argument("--out_root", type=str, default="./train_data",
                        help="Root output directory for split data")

    # normal 비율 (train/val/test 합 1.0)
    parser.add_argument("--normal_train_ratio", type=float, default=0.7)
    parser.add_argument("--normal_val_ratio", type=float, default=0.15)
    parser.add_argument("--normal_test_ratio", type=float, default=0.15)

    # defect 비율 (train=0, val/test 합 1.0)
    parser.add_argument("--defect_val_ratio", type=float, default=0.3)
    parser.add_argument("--defect_test_ratio", type=float, default=0.7)

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits")

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()

    # defect 비율 검증: 0.0 + val + test == 1.0 이어야 함
    assert abs(0.0 + opt.defect_val_ratio + opt.defect_test_ratio - 1.0) < 1e-6, \
        "defect_val_ratio + defect_test_ratio must be 1.0"

    # normal 비율 검증
    assert abs(opt.normal_train_ratio + opt.normal_val_ratio +
               opt.normal_test_ratio - 1.0) < 1e-6, \
        "normal_train_ratio + normal_val_ratio + normal_test_ratio must be 1.0"

    main(opt)
