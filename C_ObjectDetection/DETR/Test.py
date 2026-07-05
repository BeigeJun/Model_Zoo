import os
import sys

_project_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from C_ObjectDetection.DETR.DETR import DETR, PretrainedDETR
from C_ObjectDetection.DETR.DataLoader import DETRDataLoader
from C_ObjectDetection.DETR.Util import get_detr_bboxes, mAP

# VOC 클래스 이름
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

COLORS = plt.cm.get_cmap('tab20', 20).colors


def draw_predictions(image_tensor, pred_boxes, pred_labels, pred_scores,
                     gt_boxes=None, gt_labels=None, threshold=0.1, save_path=None):
    """
    이미지에 예측 박스(빨강)와 GT 박스(초록)를 시각화.
    image_tensor: (C, H, W), 정규화된 텐서
    pred_boxes: (Q, 4) [cx, cy, w, h] 0~1
    """
    # 정규화 역변환
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * std + mean).clip(0, 1)

    H, W = img.shape[:2]
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    # GT 박스 (초록)
    if gt_boxes is not None and gt_labels is not None:
        for box, label in zip(gt_boxes, gt_labels):
            if label < 0:
                continue
            cx, cy, w, h = box
            x1 = (cx - w / 2) * W
            y1 = (cy - h / 2) * H
            rect = patches.Rectangle((x1, y1), w * W, h * H,
                                      linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 4, f'GT:{VOC_CLASSES[label]}',
                    color='lime', fontsize=7, backgroundcolor='black')

    # 예측 박스 (빨강)
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        if score < threshold:
            continue
        cx, cy, w, h = box
        x1 = (cx - w / 2) * W
        y1 = (cy - h / 2) * H
        color = COLORS[label % 20]
        rect = patches.Rectangle((x1, y1), w * W, h * H,
                                  linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 4, f'{VOC_CLASSES[label]} {score:.2f}',
                color='white', fontsize=7, backgroundcolor=color)

    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장됨: {save_path}")
    else:
        plt.show()
    plt.close()


def test(model_path, use_pretrained=True, num_samples=10, threshold=0.1,
         save_images=True, calc_map=True):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.normpath(os.path.join(current_dir, '..', 'Util', 'config.yaml'))
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    NUM_CLASSES = config.get('num_class', 20)
    IMAGE_SIZE  = config.get('DETR_IMAGE_SIZE', 800)
    NUM_QUERIES = 100

    # ── 모델 로드 ──────────────────────────────────────────────────────────────
    if use_pretrained:
        model = PretrainedDETR(num_classes=NUM_CLASSES, num_queries=NUM_QUERIES).to(device)
    else:
        model = DETR(num_classes=NUM_CLASSES, num_queries=NUM_QUERIES).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"모델 로드 완료: {model_path}")

    # ── 데이터셋 ───────────────────────────────────────────────────────────────
    # testcsvfile_path가 너무 적으면 traincsvfile_path로 교체 가능
    csv_path = config['traincsvfile_path']
    test_set = DETRDataLoader(
        csv_file=csv_path,
        img_dir=config['IMG_DIR'],
        label_dir=config['LABEL_DIR'],
        image_size=IMAGE_SIZE,
        max_objects=NUM_QUERIES,
        transform='test',
    )

    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images, dim=0)
        boxes  = torch.stack([t['boxes']  for t in targets], dim=0)
        labels = torch.stack([t['labels'] for t in targets], dim=0)
        num_objects = torch.tensor([t['num_objects'] for t in targets])
        return images, {'boxes': boxes, 'labels': labels, 'num_objects': num_objects}

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=0
    )

    # ── 결과 저장 폴더 ─────────────────────────────────────────────────────────
    result_dir = os.path.join(config['save_path'], 'DETR_Test_Results')
    os.makedirs(result_dir, exist_ok=True)

    # ── 시각화 (num_samples 장) ────────────────────────────────────────────────
    print(f"\n샘플 {num_samples}장 시각화 중...")
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            if i >= num_samples:
                break

            images = images.to(device)
            pred_logits, pred_boxes = model(images)

            prob   = pred_logits[0].softmax(-1)                    # (Q, C+1)
            scores, labels = prob[:, :NUM_CLASSES].max(-1)         # (Q,)

            gt_boxes  = targets['boxes'][0].cpu().numpy()
            gt_labels = targets['labels'][0].cpu().numpy()
            n_obj     = targets['num_objects'][0].item()

            save_path = os.path.join(result_dir, f'result_{i:04d}.png') if save_images else None
            draw_predictions(
                image_tensor=images[0].cpu(),
                pred_boxes=pred_boxes[0].cpu().numpy(),
                pred_labels=labels.cpu().numpy(),
                pred_scores=scores.cpu().numpy(),
                gt_boxes=gt_boxes[:n_obj],
                gt_labels=gt_labels[:n_obj],
                threshold=threshold,
                save_path=save_path,
            )

    # ── mAP 계산 ──────────────────────────────────────────────────────────────
    if calc_map:
        print("\n전체 테스트셋 mAP 계산 중...")
        pred_boxes_all, gt_boxes_all = get_detr_bboxes(
            test_loader, model, threshold=threshold, device=device, num_classes=NUM_CLASSES
        )
        test_mAP = mAP(pred_boxes_all, gt_boxes_all, iou_threshold=0.5, num_classes=NUM_CLASSES)
        print(f"\n[결과] Test mAP@0.5 = {test_mAP:.4f} ({test_mAP * 100:.2f}%)")

        with open(os.path.join(result_dir, 'test_result.txt'), 'w') as f:
            f.write(f"Model: {model_path}\n")
            f.write(f"Test mAP@0.5: {test_mAP:.4f} ({test_mAP * 100:.2f}%)\n")
            f.write(f"Threshold: {threshold}\n")
            f.write(f"Samples visualized: {num_samples}\n")


if __name__ == '__main__':
    # 저장된 모델 경로 (Best_Accuracy_Train.pth 또는 Best_Accuracy_Validation.pth)
    MODEL_PATH = 'D:/0. Model_Save_Folder/Best_Accuracy_Validation.pth'

    # ※ annotations_Test.csv가 32장뿐이므로 num_samples는 최대 32
    #   더 많은 샘플을 보려면 Test.py 안의 csv_path를 아래처럼 변경:
    #   csv_path = config['traincsvfile_path']
    test(
        model_path=MODEL_PATH,
        use_pretrained=True,   # DETR.py의 USE_PRETRAINED와 동일하게 맞춰야 함
        num_samples=1000,        # 시각화할 이미지 수
        threshold=0.3,         # confidence 임계값
        save_images=True,      # True: 파일 저장 / False: 화면 출력
        calc_map=True,         # 전체 테스트셋 mAP 계산 여부
    )
