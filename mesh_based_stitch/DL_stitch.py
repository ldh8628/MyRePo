import os, random, cv2 as cv
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import kornia as K
from torch.utils.data import Dataset, DataLoader

# ------------ 실행 스위치 ------------
DO_TRAIN_AND_INFER = True  # True: 학습 후 DL 추론 / False: 전통식 스티칭

# ------------ 경로 ------------
IMG_DIR = "images"
IMG1 = os.path.join(IMG_DIR, "left.jpg")
IMG2 = os.path.join(IMG_DIR, "right.jpg")
CKPT_DIR = "ckpts"
CKPT_PATH = os.path.join(CKPT_DIR, "homography_net.pth")
OUT_PANO = "pano.png"
OUT_ALIGNED = "alignedB.png"

# ------------ 유틸 ------------
def imread_rgb(p):
    img = cv.imread(p, cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

def to_tensor(img):
    t = torch.from_numpy(img).float() / 255.0
    if t.ndim == 2:
        t = t[None]
    else:
        t = t.permute(2, 0, 1)
    return t

def to_np(t):
    return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

def feather_blend(base, over, mask):
    k = 21
    mb = cv.GaussianBlur(mask.astype(np.float32), (k, k), 0)
    mb = np.clip(mb, 1e-6, 1.0)
    out = over * mb[..., None] + base * (1 - mb[..., None])
    return out

def warp_persp(img_t, H, h, w):
    return K.geometry.transform.warp_perspective(
        img_t, H, dsize=(h, w), mode="bilinear",
        padding_mode="zeros", align_corners=True
    )

def overlap_mask(h, w, H, device):
    """
    워핑에서 유효한(소스에서 샘플링 가능한) 영역 마스크
    핵심 변경:
      - H가 아니라 H^{-1}로 유효성 체크(워핑 샘플링 방향과 일치)
    """
    grid = K.utils.create_meshgrid(h, w, normalized_coordinates=True)[0].to(device)
    x = (grid[..., 0] + 1) * (w - 1) / 2
    y = (grid[..., 1] + 1) * (h - 1) / 2
    ones = torch.ones_like(x)
    pts = torch.stack([x, y, ones], 0).reshape(3, -1)  # [3, HW]

    Hinv = torch.inverse(H)
    Hp = Hinv @ pts
    xp = Hp[0] / (Hp[2] + 1e-8)
    yp = Hp[1] / (Hp[2] + 1e-8)

    m = ((xp >= 0) & (xp < w) & (yp >= 0) & (yp < h)).reshape(h, w).float()
    return m

# ------------ 모델 ------------
class CorrBlock(nn.Module):
    def forward(self, f1, f2):
        n, c, h, w = f1.shape
        f1 = F.normalize(f1, 1)
        f2 = F.normalize(f2, 1)
        r = 4
        outs = []
        f2p = F.pad(f2, (r, r, r, r), mode="replicate")
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                outs.append((f1 * f2p[:, :, dy + r : dy + r + h, dx + r : dx + r + w]).sum(1, True))
        return torch.cat(outs, 1)

class HomographyNet(nn.Module):
    def __init__(self):
        super().__init__()

        def feat():
            return nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(True),
                nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(True),
                nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(True),
                nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(True),
            )

        self.fA = feat()
        self.fB = feat()
        self.corr = CorrBlock()

        ch = (2 * 4 + 1) ** 2  # 81
        self.head = nn.Sequential(
            nn.Conv2d(ch, 128, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(True),
            nn.Linear(128, 8),
        )

    def forward(self, A, B):
        c = self.corr(self.fA(A), self.fB(B))
        p = self.head(c)

        # 핵심 변경: 파라미터 폭주 방지(최소 수정)
        p = 0.1 * torch.tanh(p)

        H = torch.zeros(A.size(0), 3, 3, device=A.device)
        H[:, 0, 0] = 1
        H[:, 1, 1] = 1
        H[:, 2, 2] = 1
        H[:, 0, 0] += p[:, 0]; H[:, 0, 1] += p[:, 1]; H[:, 0, 2] += p[:, 2]
        H[:, 1, 0] += p[:, 3]; H[:, 1, 1] += p[:, 4]; H[:, 1, 2] += p[:, 5]
        H[:, 2, 0] += p[:, 6]; H[:, 2, 1] += p[:, 7]
        return H

# ------------ 합성 데이터셋(이미지 2장만으로 무한 샘플) ------------
class SynthHomographyPairs(Dataset):
    def __init__(self, img_paths, size=320, steps=2000, batch=12, max_pix=0.18):
        self.imgs = [imread_rgb(p) for p in img_paths]
        if len(self.imgs) == 0:
            raise RuntimeError("이미지 없음")
        self.size = size
        self.n = steps * batch
        self.max_pix = max_pix

    def __len__(self):
        return self.n

    def rand_homography(self, h, w):
        off = max(w, h) * self.max_pix
        pts1 = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        jitter = np.random.uniform(-off, off, size=(4, 2)).astype(np.float32)
        pts2 = pts1 + jitter
        H, _ = cv.findHomography(pts2, pts1, 0)  # img2 -> img1
        return H

    def __getitem__(self, idx):
        img = random.choice(self.imgs)
        img = cv.resize(img, (self.size, self.size), interpolation=cv.INTER_AREA)
        h, w = img.shape[:2]

        H = self.rand_homography(h, w)
        img2 = cv.warpPerspective(img, H, (w, h), flags=cv.INTER_LINEAR)

        # 핵심 변경: 초기 수렴 방해 가능성이 큰 밝기/바이어스 증강 제거(최소 수정)
        # g = np.clip(np.random.normal(1.0, 0.08), 0.8, 1.2)
        # b = np.random.uniform(-0.05, 0.05)
        # img2 = np.clip(img2.astype(np.float32) * g + b * 255, 0, 255).astype(np.uint8)

        A = to_tensor(img)
        B = to_tensor(img2)
        return A, B

# ------------ 손실 ------------
def photometric_ssim_L1(x, y, mask=None):
    """
    x, y: [B,C,H,W] in [0,1]
    mask: [B,H,W] in [0,1]
    """
    ssim_loss = K.losses.SSIMLoss(window_size=5)(x, y)  # 이미 loss

    l1 = (x - y).abs()

    if mask is not None:
        m = mask[:, None].clamp(0, 1)  # [B,1,H,W]

        # SSIMLoss 출력이 [B]일 수도 있어, 배치 가중치로만 마스크 반영(최소 수정 + 안전)
        w = m.mean(dim=(2, 3)).squeeze(1).clamp_min(1e-6)  # [B]
        if hasattr(ssim_loss, "ndim") and ssim_loss.ndim > 0:
            ssim_loss = (ssim_loss * w).sum() / w.sum()
        # scalar면 그대로 사용

        l1 = (l1 * m).sum() / m.sum().clamp_min(1.0)
    else:
        if hasattr(ssim_loss, "ndim") and ssim_loss.ndim > 0:
            ssim_loss = ssim_loss.mean()
        l1 = l1.mean()

    return 0.85 * ssim_loss + 0.15 * l1

def smoothness_reg(H):
    g, h = H[:, 2, 0], H[:, 2, 1]
    return (g**2 + h**2).mean() * 0.01

# ------------ 학습 ------------
def train_short(img_paths, ckpt_path, size=320, steps=1200, batch=12, lr=1e-4):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    ds = SynthHomographyPairs(img_paths, size=size, steps=steps, batch=batch)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = HomographyNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    net.train()
    step = 0
    for A, B in dl:
        A, B = A.to(device), B.to(device)

        H = net(A, B)
        Bw = warp_persp(B, H, A.size(2), A.size(3))

        # 핵심 변경: mask shape 꼬임 방지 (unsqueeze 제거)
        m_list = []
        for i in range(A.size(0)):
            m_list.append(overlap_mask(A.size(2), A.size(3), H[i], device))
        M = torch.stack(m_list, 0).to(device)  # [B,H,W]

        loss = photometric_ssim_L1(A, Bw, mask=M) + smoothness_reg(H)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)  # 폭주 방지(안전한 최소 추가)
        opt.step()

        step += 1
        if step % 50 == 0:
            print(f"[train {step}] loss={loss.item():.4f}")
        if step >= steps:
            break

    torch.save({"state": net.state_dict()}, ckpt_path)
    print("saved ckpt:", ckpt_path)
    return ckpt_path

# ------------ 추론(DL) ------------
def infer_dl(A0, B0, ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = HomographyNet().to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device)["state"])
    net.eval()

    A = to_tensor(A0)[None].to(device)
    B = to_tensor(B0)[None].to(device)

    with torch.no_grad():
        H = net(A, B)
        Bw = warp_persp(B, H, A.size(2), A.size(3))

    A_np, Bw_np = to_np(A[0]), to_np(Bw[0])

    M = overlap_mask(A.size(2), A.size(3), H[0], device).cpu().numpy()
    M = cv.GaussianBlur((M > 0).astype(np.float32), (9, 9), 0)

    blend = feather_blend(A_np.astype(np.float32) / 255.0, Bw_np.astype(np.float32) / 255.0, M)
    pano = (blend * 255).astype(np.uint8)

    cv.imwrite(OUT_PANO, cv.cvtColor(pano, cv.COLOR_RGB2BGR))
    cv.imwrite(OUT_ALIGNED, cv.cvtColor(Bw_np, cv.COLOR_RGB2BGR))
    print("[DL] saved:", OUT_PANO, OUT_ALIGNED)

# ------------ 전통식 경로 ------------
def classical_stitch(A_rgb, B_rgb):
    A = cv.cvtColor(A_rgb, cv.COLOR_RGB2BGR)
    B = cv.cvtColor(B_rgb, cv.COLOR_RGB2BGR)

    if hasattr(cv, "SIFT_create"):
        det = cv.SIFT_create()
        norm = cv.NORM_L2
    else:
        det = cv.ORB_create(5000)
        norm = cv.NORM_HAMMING

    k1, d1 = det.detectAndCompute(A, None)
    k2, d2 = det.detectAndCompute(B, None)
    if d1 is None or d2 is None:
        raise RuntimeError("특징 없음")

    m = cv.BFMatcher(norm).knnMatch(d1, d2, k=2)
    good = [a for a, b in m if a.distance < 0.75 * b.distance]
    if len(good) < 20:
        raise RuntimeError("매칭 부족")

    src = np.float32([k1[g.queryIdx].pt for g in good])
    dst = np.float32([k2[g.trainIdx].pt for g in good])
    H, _ = cv.findHomography(dst, src, cv.RANSAC, 3.0)

    h, w = A.shape[:2]
    Bw = cv.warpPerspective(B, H, (w, h))
    mask = (Bw.sum(2) > 0).astype(np.float32)

    blend = feather_blend(A.astype(np.float32) / 255.0, Bw.astype(np.float32) / 255.0, mask)
    pano = (blend * 255).astype(np.uint8)

    cv.imwrite(OUT_PANO, pano)
    print("[CLASSICAL] saved:", OUT_PANO)

# ------------ 메인 ------------
if __name__ == "__main__":
    A0 = imread_rgb(IMG1)
    B0 = imread_rgb(IMG2)

    target_h = max(480, min(1080, A0.shape[0]))
    s = target_h / A0.shape[0]
    rs = lambda I: cv.resize(I, (int(I.shape[1] * s), int(I.shape[0] * s)), interpolation=cv.INTER_AREA)
    A0, B0 = rs(A0), rs(B0)

    if DO_TRAIN_AND_INFER:
        ck = train_short([IMG1, IMG2], CKPT_PATH, size=320, steps=1200, batch=12, lr=1e-4)
        infer_dl(A0, B0, ck)
    else:
        classical_stitch(A0, B0)
