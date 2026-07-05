import os
import sys

_project_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DETRDataLoader(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, image_size=800, max_objects=100, transform='train'):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.max_objects = max_objects

        self.train_transforms = A.Compose(
            [
                A.LongestMaxSize(max_size=image_size),
                A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.4),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="yolo", min_visibility=0.3, label_fields=[]),
        )

        self.test_transforms = A.Compose(
            [
                A.LongestMaxSize(max_size=image_size),
                A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="yolo", min_visibility=0.3, label_fields=[]),
        )

        self.transform = self.train_transforms if transform == 'train' else self.test_transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])

        # [class_id, cx, cy, w, h] -> albumentations yolo 형식: [cx, cy, w, h, class_id]
        raw = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2)
        bboxes = np.roll(raw, 4, axis=1).tolist()  # class 맨 뒤로

        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            aug = self.transform(image=image, bboxes=bboxes)
            image = aug["image"]
            bboxes = aug["bboxes"]

        # DETR 타겟: boxes (N, 4) in [cx, cy, w, h], labels (N,)
        # 패딩으로 max_objects 크기 고정
        boxes = torch.zeros((self.max_objects, 4), dtype=torch.float32)
        labels = torch.full((self.max_objects,), fill_value=-1, dtype=torch.long)

        num_obj = min(len(bboxes), self.max_objects)
        for i in range(num_obj):
            cx, cy, w, h, cls = bboxes[i]
            boxes[i] = torch.tensor([cx, cy, w, h])
            labels[i] = int(cls)

        target = {
            "boxes": boxes,           # (max_objects, 4)
            "labels": labels,         # (max_objects,)
            "num_objects": num_obj,   # 실제 객체 수
        }

        return image, target
