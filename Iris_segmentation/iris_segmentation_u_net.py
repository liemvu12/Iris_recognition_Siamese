import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchmetrics
from torchmetrics import Dice, JaccardIndex
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2 # np.array -> torch.tensor
import os
from tqdm import tqdm
from glob import glob
from main import *
class IrisDataset(Dataset):
    def __init__(self, root_dir, ROI_input, transform=None):
        self.root_dir = root_dir
        self.ROI_input = ROI_input
        self.transform = transform
        self.img_path_lst = []
        img_files = [f for f in os.listdir(self.ROI_input) if f.endswith(".jpg")]
        for filename in img_files:
            self.img_path_lst.append(filename)
    def __len__(self):
        return len(self.img_path_lst)
    def __getitem__(self, idx):
        image_path = os.path.join(self.ROI_input, self.img_path_lst[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=image)
            transformed_image = transformed['image']
            return transformed_image
        else:
            return image

trainsize = 384
test_trainsform = A.Compose([
    A.Resize(width=trainsize, height=trainsize),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(),
])
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
def unet_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        nn.ReLU()
    )
class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.block_down1 = unet_block(3, 64)
        self.block_down2 = unet_block(64, 128)
        self.block_down3 = unet_block(128, 256)
        self.block_down4 = unet_block(256, 512)
        self.block_neck = unet_block(512, 1024)
        self.block_up1 = unet_block(1024+512, 512)
        self.block_up2 = unet_block(256+512, 256)
        self.block_up3 = unet_block(128+256, 128)
        self.block_up4 = unet_block(128+64, 64)
        self.conv_cls = nn.Conv2d(64, self.n_classes, 1)
    def forward(self, x):
        # (B, C, H, W)
        x1 = self.block_down1(x)
        x = self.downsample(x1)
        x2 = self.block_down2(x)
        x = self.downsample(x2)
        x3 = self.block_down3(x)
        x = self.downsample(x3)
        x4 = self.block_down4(x)
        x = self.downsample(x4)
        x = self.block_neck(x)
        x = torch.cat([x4, self.upsample(x)], dim=1)
        x = self.block_up1(x)
        x = torch.cat([x3, self.upsample(x)], dim=1)
        x = self.block_up2(x)
        x = torch.cat([x2, self.upsample(x)], dim=1)
        x = self.block_up3(x)
        x = torch.cat([x1, self.upsample(x)], dim=1)
        x = self.block_up4(x)
        x = self.conv_cls(x)
        return x

model = UNet(1)
model.load_state_dict(torch.load('model_ep_30.pth', map_location=torch.device('cpu')))
model.eval()

sourceFileDir = os.path.dirname(os.path.abspath(__file__))
test_dataset = IrisDataset(sourceFileDir, 'data_iris/from_detec', test_trainsform)

device = torch.device('cpu')
model.eval()
with torch.no_grad():
    x = test_dataset[0]
    x = x.to(device).float().unsqueeze(0)
    y_hat = model(x).squeeze()
    y_hat_mask = y_hat.sigmoid().round().long()

    # Chuyển tensor thành mảng NumPy và hiển thị hình ảnh
    img = unorm(x.squeeze().cpu()).permute(1, 2, 0).numpy()
    mask = y_hat_mask.cpu().numpy()

    # Scale lại mask từ [0, 1] thành [0, 255] để hiển thị bằng OpenCV
    mask = (mask * 255).astype(np.uint8)
    mask = np.stack([mask] * 3, axis=-1)
    mask = mask / 255.0
    mask = mask.astype('float32')

    blurred_mask = blur_white_mask(mask)
    overlaid_img = overlay_images(img, blurred_mask)
    output = process_image(overlaid_img)

    cv2.imshow('iris', img)
    cv2.imshow('iris_seg',output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()