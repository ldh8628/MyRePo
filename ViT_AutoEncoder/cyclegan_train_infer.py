import os
import argparse
import itertools
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


# ---------------------------------------
# 1. Dataset
# ---------------------------------------
class ImageDataset(Dataset):
    def __init__(self, rootA, rootB, transform):
        self.transform = transform
        self.files_A = sorted([os.path.join(rootA, f) for f in os.listdir(rootA)
                               if f.lower().endswith(("jpg", "jpeg", "png"))])
        self.files_B = sorted([os.path.join(rootB, f) for f in os.listdir(rootB)
                               if f.lower().endswith(("jpg", "jpeg", "png"))])

        self.lenA = len(self.files_A)
        self.lenB = len(self.files_B)
        self.max_len = max(self.lenA, self.lenB)

    def __len__(self):
        return self.max_len

    def __getitem__(self, idx):
        imgA = Image.open(self.files_A[idx % self.lenA]).convert("RGB")
        imgB = Image.open(self.files_B[idx % self.lenB]).convert("RGB")

        return {
            "A": self.transform(imgA),
            "B": self.transform(imgB)
        }


# ---------------------------------------
# 2. Generator & Discriminator
# ---------------------------------------
class ResidualBlock(nn.Module):
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


class Generator(nn.Module):
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
            model += [ResidualBlock(in_features)]

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


class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super().__init__()
        model = [
            nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        ]

        in_features = 64
        out_features = in_features * 2
        for _ in range(3):
            model += [
                nn.Conv2d(in_features, out_features, 4, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.LeakyReLU(0.2),
            ]
            in_features = out_features
            out_features = in_features * 2

        model += [nn.Conv2d(in_features, 1, 4, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# ---------------------------------------
# 3. Training Function
# ---------------------------------------
def train(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    dataset = ImageDataset(opt.normal, opt.defect, transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Generators
    G_AB = Generator().to(device)
    G_BA = Generator().to(device)

    # Discriminators
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(0.5, 0.999)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    for epoch in range(opt.epochs):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{opt.epochs}")
        for batch in loop:
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            # --------------------
            # Train G
            # --------------------
            optimizer_G.zero_grad()

            # Identity Loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) * 5.0

            # GAN Loss (A -> B)
            fake_B = G_AB(real_A)
            pred_fake_B = D_B(fake_B)
            valid_B_for_G = torch.ones_like(pred_fake_B)
            loss_GAN_AB = criterion_GAN(pred_fake_B, valid_B_for_G)

            # GAN Loss (B -> A)
            fake_A = G_BA(real_B)
            pred_fake_A = D_A(fake_A)
            valid_A_for_G = torch.ones_like(pred_fake_A)
            loss_GAN_BA = criterion_GAN(pred_fake_A, valid_A_for_G)

            # Cycle Loss
            recov_A = G_BA(fake_B)
            recov_B = G_AB(fake_A)
            loss_cycle = (criterion_cycle(recov_A, real_A) +
                          criterion_cycle(recov_B, real_B)) * 10.0

            loss_G = loss_identity + loss_GAN_AB + loss_GAN_BA + loss_cycle
            loss_G.backward()
            optimizer_G.step()

            # --------------------
            # Train D_A
            # --------------------
            optimizer_D_A.zero_grad()
            # Real
            pred_real_A = D_A(real_A)
            valid_A_for_D = torch.ones_like(pred_real_A)
            loss_real_A = criterion_GAN(pred_real_A, valid_A_for_D)
            # Fake
            pred_fake_A = D_A(fake_A.detach())
            fake_A_for_D = torch.zeros_like(pred_fake_A)
            loss_fake_A = criterion_GAN(pred_fake_A, fake_A_for_D)
            loss_D_A = (loss_real_A + loss_fake_A) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()

            # --------------------
            # Train D_B
            # --------------------
            optimizer_D_B.zero_grad()
            # Real
            pred_real_B = D_B(real_B)
            valid_B_for_D = torch.ones_like(pred_real_B)
            loss_real_B = criterion_GAN(pred_real_B, valid_B_for_D)
            # Fake
            pred_fake_B_for_D = D_B(fake_B.detach())
            fake_B_for_D = torch.zeros_like(pred_fake_B_for_D)
            loss_fake_B = criterion_GAN(pred_fake_B_for_D, fake_B_for_D)
            loss_D_B = (loss_real_B + loss_fake_B) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()

            loop.set_postfix(G=loss_G.item(),
                             DA=loss_D_A.item(), DB=loss_D_B.item())

        torch.save(G_AB.state_dict(), "G_AB.pth")
        torch.save(G_BA.state_dict(), "G_BA.pth")

    print("Training finished.")


# ---------------------------------------
# 4. Inference (Normal → Defect)
# ---------------------------------------
def infer(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G_AB = Generator().to(device)
    G_AB.load_state_dict(torch.load(opt.model, map_location=device))
    G_AB.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    os.makedirs(opt.out, exist_ok=True)

    for fname in os.listdir(opt.normal):
        if not fname.lower().endswith(("jpg", "jpeg", "png")):
            continue

        img_path = os.path.join(opt.normal, fname)
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            fake_B = G_AB(x)[0]

        out = fake_B.cpu().numpy().transpose(1, 2, 0)
        out = ((out * 0.5 + 0.5) * 255).astype(np.uint8)

        save_path = os.path.join(opt.out, fname)
        cv2.imwrite(save_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    print("Inference done.")


# ---------------------------------------
# 5. Main
# ---------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--normal", type=str, default="./data/normal")
    parser.add_argument("--defect", type=str, default="./data/defect")
    parser.add_argument("--model", type=str, default="G_AB.pth")
    parser.add_argument("--out", type=str, default="./generated_defects")
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--epochs", type=int, default=50)

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--infer", action="store_true")

    opt = parser.parse_args()

    if opt.train:
        train(opt)
    elif opt.infer:
        infer(opt)
    else:
        print("Please specify either --train or --infer")
