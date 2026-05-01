# standard library
import os
import math
import random
import csv
import re
import shutil
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from typing import (
    List, Tuple, Any, Iterator, Dict, Optional,
    Iterable, Union, Sequence
)
from urllib.parse import urlparse, unquote

# third-party
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as T

# augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# lightning
import pytorch_lightning as pl


train_transform = A.Compose([
    A.OneOf([
        A.Blur(blur_limit=[2,5], p=0.5),
        A.CLAHE(clip_limit=(1.1, 1.2), p=0.5),
    ], p=0.3),
    A.ImageCompression(quality_range=[60, 100], p=0.25),
    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), 
                               contrast_limit=(-0.1, 0.1), p=0.2),
    A.ToGray(num_output_channels=1, p=1.0),

    ToTensorV2(),
])

val_transform = A.Compose([
    A.ToGray(num_output_channels=1, p=1.0),
    ToTensorV2(),
])



class SobelTransform:
    def __call__(self, img):
        if isinstance(img, Image.Image):
            gray = np.asarray(img.convert("L"))
        else:
            arr = np.asarray(img)
            if arr.ndim == 3:
                # RGB -> grayscale
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            else:
                gray = arr

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx ** 2 + gy ** 2)
        mx = grad.max() if grad.max() != 0 else 1.0
        grad = (grad / mx * 255.0).astype(np.uint8)

        grad3 = np.stack([grad, grad, grad], axis=-1)  # H x W x 3 uint8
        return grad3

sobel_transform = SobelTransform()

class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding='same', bias=False)

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x

class TextCropDataset(Dataset):
    def __init__(self, triples: List[Tuple[Any, float, float]], transform=None, use_sobel=False):
        self.triples = triples
        self.transform = transform  # albumentations transform
        self.use_sobel = use_sobel
        self.sobel_func = Sobel()

    def __len__(self):
        return len(self.triples)

    def _load_img_for_albu(self, img_src):
        if isinstance(img_src, str):
            pil = Image.open(img_src).convert("RGB")
            img_np = np.asarray(pil)
        elif isinstance(img_src, Image.Image):
            img_np = np.asarray(img_src.convert("RGB"))
        elif isinstance(img_src, torch.Tensor):
            t = img_src.detach().cpu()
            if t.ndim == 3:
                t = t.permute(1,2,0)  # HWC
            img_np = (t.numpy() * 255.0).astype(np.uint8)
        elif isinstance(img_src, np.ndarray):
            img_np = img_src
            if img_np.ndim == 2:
                img_np = np.stack([img_np]*3, axis=-1)
        else:
            raise RuntimeError(f"Unsupported image type: {type(img_src)}")
        return img_np

    def __getitem__(self, idx):
        path_or_img, ocr_score, rating = self.triples[idx]
        img_np = self._load_img_for_albu(path_or_img)

        if self.transform is not None:
            out = self.transform(image=img_np)
            img = out["image"] 
            
            if isinstance(img, torch.Tensor):
                if img.dtype == torch.uint8:
                    img = img.float().div(255.0)
                else:
                    img = img.float()
            else:
                img = transforms.ToTensor()(Image.fromarray(img))
        else:
            img = transforms.ToTensor()(Image.fromarray(img_np))

        if self.use_sobel:
            togray = transforms.Grayscale(num_output_channels=1)
            img = togray(img)
            sobel = self.sobel_func(img.unsqueeze(0)).squeeze(0)
            img = torch.cat((img, sobel), dim=0) 
            

        ocr_score = torch.tensor(float(ocr_score), dtype=torch.float32)
        rating = torch.tensor(float(rating), dtype=torch.float32)
            
        return img, ocr_score, rating

def pad_collate_center(batch,
                       pad_value: float = 0.0,
                       return_mask: bool = False,
                       target_h: int = 200,
                       pad_to: int = None,
                       max_width: int = None):
    imgs, ocrs, ratings = zip(*batch)
    B = len(imgs)
    C = imgs[0].shape[0]

    device = imgs[0].device if imgs[0].is_cuda else torch.device("cpu")

    resized_imgs = []
    widths = []
    heights = []

    for img in imgs:
        img_f = img.float()

        c, h, w = int(img_f.shape[0]), int(img_f.shape[1]), int(img_f.shape[2])
        heights.append(h); widths.append(w)

        if h == target_h:
            img_rs = img_f
        else:
            scale = float(target_h) / float(h)
            new_w = max(1, int(round(w * scale)))
            img_rs = F.interpolate(img_f.unsqueeze(0), size=(target_h, new_w), mode='bilinear', align_corners=False)
            img_rs = img_rs.squeeze(0)
            
        # apply max_width clamp by center-cropping if needed
        if (max_width is not None) and (img_rs.shape[2] > max_width):
            cur_w = img_rs.shape[2]
            left = (cur_w - max_width) // 2
            img_rs = img_rs[:, :, left:left + max_width]
        resized_imgs.append(img_rs)
        widths[-1] = int(img_rs.shape[2])  # update width after possible crop
        heights[-1] = int(img_rs.shape[1])

    # compute target width for batch
    max_w = max(widths)
    if pad_to is not None and pad_to > 0:
        max_w = int(math.ceil(max_w / pad_to) * pad_to)

    H = target_h
    W = max_w
    C = resized_imgs[0].shape[0]

    batch_imgs = torch.full((B, C, H, W), float(pad_value), dtype=torch.float32, device=device)
    mask = torch.zeros((B, 1, H, W), dtype=torch.bool, device=device)

    for i, img in enumerate(resized_imgs):
        c, h, w = img.shape
        top = (H - h) // 2
        left = (W - w) // 2
        batch_imgs[i, :c, top:top + h, left:left + w] = img.to(device=device, dtype=torch.float32)
        mask[i, 0, top:top + h, left:left + w] = True

    ocrs = torch.stack([o.view(1).float() for o in ocrs], dim=0)
    ratings = torch.stack([r.view(1).float() for r in ratings], dim=0)

    if return_mask:
        return batch_imgs, ocrs, ratings, mask
    return batch_imgs, ocrs, ratings

def collate_preserve_single(batch):
    if len(batch) != 1:
        return pad_collate_center(batch, pad_value=0.0, return_mask=False)

    img, ocr, rating = batch[0]

    if not isinstance(img, torch.Tensor):
        img = transforms.ToTensor()(Image.fromarray(img))
    if img.dtype == torch.uint8:
        img = img.float().div(255.0)
    else:
        img = img.float()

    img_b = img.unsqueeze(0)             
    ocr_b = ocr.view(1, 1).float()
    rating_b = rating.view(1, 1).float()
    return img_b, ocr_b, rating_b

def infer_size_from_src(src) -> Tuple[int,int]:
    if isinstance(src, np.ndarray):
        if src.ndim == 2:
            h, w = src.shape
        elif src.ndim == 3:
            h, w = src.shape[0], src.shape[1] if src.shape[2] in (3,4) else (src.shape[0], src.shape[1])
            # careful: assume HxWxC
            h, w = src.shape[0], src.shape[1]
        else:
            raise RuntimeError("Unsupported numpy shape: " + str(src.shape))
    elif isinstance(src, Image.Image):
        w, h = src.size
    elif isinstance(src, torch.Tensor):
        if src.ndim == 3:
            # try CxHxW or HxWxC: detect
            if src.shape[0] in (1,3,4):
                h, w = int(src.shape[1]), int(src.shape[2])
            else:
                h, w = int(src.shape[0]), int(src.shape[1])
        else:
            raise RuntimeError("Unsupported tensor shape: " + str(src.shape))
    elif isinstance(src, str):
        # assume path to image — open minimally
        with Image.open(src) as im:
            w, h = im.size
    else:
        raise RuntimeError(f"Unsupported image source type: {type(src)}")
    return int(h), int(w)


class TextQualityDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_triples,
                 test_triples,
                 val_size=50,
                 batch_size=32,
                 batch_size_test=4,
                 num_workers=8,
                 train_transform=None,
                 val_transform=None,
                 use_sobel=False,
                 preserve_single_image: bool = False):
        super().__init__()
        assert len(train_triples) > val_size, "train_triples must be larger than val_size"
        self.train_triples = train_triples
        self.test_triples = test_triples
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.use_sobel = use_sobel
        self.batch_size_test = batch_size_test

        # placeholders
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.preserve_single_image = preserve_single_image  

    def setup(self, stage=None):
        val_triples = self.train_triples[:self.val_size]
        train_triples = self.train_triples[self.val_size:]
        self.train_ds = TextCropDataset(train_triples, transform=self.train_transform, use_sobel=self.use_sobel)
        self.val_ds = TextCropDataset(val_triples, transform=self.val_transform, use_sobel=self.use_sobel)
        self.test_ds = TextCropDataset(self.test_triples, transform=self.val_transform, use_sobel=self.use_sobel)

    def train_dataloader(self):
        if self.preserve_single_image:
            return DataLoader(self.train_ds,
                              batch_size=1,
                              shuffle=True,  
                              num_workers=self.num_workers,
                              collate_fn=collate_preserve_single,
                              pin_memory=True)
        
        return DataLoader(self.train_ds,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.num_workers,
                            collate_fn=lambda b: pad_collate_center(b, pad_value=0.0),
                            pin_memory=True)

    def val_dataloader(self):
        if self.preserve_single_image:
            return DataLoader(self.val_ds,
                              batch_size=1,
                              shuffle=True,
                              num_workers=self.num_workers,
                              collate_fn=collate_preserve_single,
                              pin_memory=True)
            
        return DataLoader(self.val_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=lambda b: pad_collate_center(b, pad_value=0.0),
                          pin_memory=True)

    def test_dataloader(self):
        if self.preserve_single_image:
            return DataLoader(self.test_ds,
                              batch_size=1,
                              shuffle=False,
                              num_workers=self.num_workers,
                              collate_fn=collate_preserve_single,
                              pin_memory=True)
        
        return DataLoader(self.test_ds,
                          batch_size=self.batch_size_test,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=lambda b: pad_collate_center(b, pad_value=0.0),
                          pin_memory=True)


