import os
import pandas as pd
from bs4 import BeautifulSoup as bs

label_set = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]

def get_coord(tag, coord, type=int):
    return float(tag.select(coord)[0].text) if type == float else int(tag.select(coord)[0].text)

def normalize(bbox, w, h):
    x_min, y_min, x_max, y_max = bbox
    center_x = ((x_max + x_min) / 2) / w
    center_y = ((y_max + y_min) / 2) / h
    width = (x_max - x_min) / w
    height = (y_max - y_min) / h
    return (center_x, center_y, width, height)

def pascal2yolo(xml_path):
    soup = bs(open(xml_path, 'r'), 'lxml')
    width = int(soup.select('size > width')[0].text)
    height = int(soup.select('size > height')[0].text)
    file_name = soup.select('filename')[0].text
    labels, bboxes = [], []
    for obj in soup.select('object'):
        name = obj.select('name')[0].text
        if name not in label_set:
            continue
        bbox_tag = obj.select('bndbox')[0]
        bbox = (
            get_coord(bbox_tag, 'xmin', float),
            get_coord(bbox_tag, 'ymin', float),
            get_coord(bbox_tag, 'xmax', float),
            get_coord(bbox_tag, 'ymax', float),
        )
        norm_bbox = normalize(bbox, width, height)
        labels.append(label_set.index(name))
        bboxes.append(norm_bbox)
    return labels, bboxes, file_name

def convert_voc_to_yolo(xml_folder, txt_folder):
    os.makedirs(txt_folder, exist_ok=True)
    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith('.xml'):
            continue
        xml_path = os.path.join(xml_folder, xml_file)
        labels, bboxes, fname = pascal2yolo(xml_path)
        if not labels:
            continue
        fname = os.path.splitext(fname)[0]
        txt_path = os.path.join(txt_folder, f'{fname}.txt')
        with open(txt_path, 'w') as f:
            for label, bbox in zip(labels, bboxes):
                bbox_str = ' '.join(map(str, bbox))
                f.write(f"{label} {bbox_str}\n")

xml_folder = 'D:/1. DataSet/2-1. Pascal/VOC2/VOC2012/VOC2012/Annotations'
label_folder = 'D:/1. DataSet/2-1. Pascal/VOC2/VOC2012/VOC2012/MakedLabels'
convert_voc_to_yolo(xml_folder, label_folder)

img_folder = 'D:/1. DataSet/2-1. Pascal/VOC2/VOC2012/VOC2012/JPEGImages'
csv_path = 'D:/1. DataSet/2-1. Pascal/VOC2/VOC2012/VOC2012/annotations.csv'

import random

img_files   = sorted([f for f in os.listdir(img_folder)   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
label_files = set([f for f in os.listdir(label_folder) if f.endswith('.txt')])

rows = []
for img in img_files:
    base, _ = os.path.splitext(img)
    label = f'{base}.txt'
    if label in label_files:
        rows.append({'image': img, 'label': label})

random.shuffle(rows)
split = int(len(rows) * 0.9)
train_rows = rows[:split]
test_rows  = rows[split:]

train_csv = 'D:/1. DataSet/2-1. Pascal/VOC2/VOC2012/VOC2012/annotations_Train.csv'
test_csv  = 'D:/1. DataSet/2-1. Pascal/VOC2/VOC2012/VOC2012/annotations_Test.csv'

pd.DataFrame(train_rows).to_csv(train_csv, index=False)
pd.DataFrame(test_rows).to_csv(test_csv,  index=False)
print(f"Train: {len(train_rows)}장, Test: {len(test_rows)}장")
df = pd.DataFrame(rows)
df.to_csv(csv_path, index=False)

