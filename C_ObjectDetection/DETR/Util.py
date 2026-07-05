import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ─── Hungarian Matching ──────────────────────────────────────────────────────

def generalized_iou(boxes1, boxes2):
    """GIoU between boxes in [cx, cy, w, h] format."""
    # [cx,cy,w,h] -> [x1,y1,x2,y2]
    b1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
    b1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
    b1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
    b1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2

    b2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
    b2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
    b2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
    b2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2

    inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
    inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
    inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))
    inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter

    iou = inter / (union + 1e-6)

    enc_x1 = torch.min(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
    enc_y1 = torch.min(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
    enc_x2 = torch.max(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))
    enc_y2 = torch.max(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))
    enc_area = (enc_x2 - enc_x1).clamp(0) * (enc_y2 - enc_y1).clamp(0)

    giou = iou - (enc_area - union) / (enc_area + 1e-6)
    return giou


class HungarianMatcher(nn.Module):
    """
    예측 쿼리와 GT 박스 사이의 최적 이분매칭(Hungarian algorithm) 수행.
    비용: classification + L1 box + GIoU
    """
    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, pred_logits, pred_boxes, targets):
        """
        pred_logits: (B, num_queries, num_classes)
        pred_boxes:  (B, num_queries, 4)  [cx,cy,w,h] sigmoid 후
        targets: list of dict with 'boxes'(N,4), 'labels'(N,), 'num_objects'(int)
        반환: list of (pred_idx, gt_idx) 튜플 (배치별)
        """
        B, Q, _ = pred_logits.shape
        indices = []

        for b in range(B):
            n = targets[b]["num_objects"]
            if n == 0:
                indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue

            gt_boxes = targets[b]["boxes"][:n].to(pred_boxes.device)
            gt_labels = targets[b]["labels"][:n].to(pred_logits.device)

            # 클래스 비용: negative log-softmax
            prob = pred_logits[b].softmax(-1)                     # (Q, C)
            cost_cls = -prob[:, gt_labels]                         # (Q, n)

            # L1 박스 비용
            cost_l1 = torch.cdist(pred_boxes[b], gt_boxes, p=1)  # (Q, n)

            # GIoU 비용
            cost_g = -generalized_iou(pred_boxes[b], gt_boxes)    # (Q, n)

            C = (self.cost_class * cost_cls
                 + self.cost_bbox * cost_l1
                 + self.cost_giou * cost_g).cpu()

            row_idx, col_idx = linear_sum_assignment(C.numpy())
            indices.append((torch.as_tensor(row_idx, dtype=torch.long),
                            torch.as_tensor(col_idx, dtype=torch.long)))

        return indices


# ─── DETR Loss ───────────────────────────────────────────────────────────────

class DETRLoss(nn.Module):
    def __init__(self, num_classes, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0,
                 lambda_cls=1.0, lambda_bbox=5.0, lambda_giou=2.0, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(cost_class, cost_bbox, cost_giou)
        self.lambda_cls = lambda_cls
        self.lambda_bbox = lambda_bbox
        self.lambda_giou = lambda_giou

        # 배경(no-object) 클래스 가중치를 낮게 설정
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, pred_logits, pred_boxes, targets):
        """
        pred_logits: (B, Q, num_classes+1)  마지막 인덱스 = no-object
        pred_boxes:  (B, Q, 4)              [cx,cy,w,h] 0~1 범위
        """
        indices = self.matcher(pred_logits, pred_boxes, targets)

        # ── 클래스 손실 ───────────────────────────────────────────────────────
        B, Q, _ = pred_logits.shape
        target_cls = torch.full((B, Q), self.num_classes, dtype=torch.long, device=pred_logits.device)

        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            gt_labels = targets[b]["labels"][gt_idx].to(pred_logits.device)
            target_cls[b, pred_idx] = gt_labels

        loss_cls = F.cross_entropy(
            pred_logits.reshape(B * Q, -1),
            target_cls.reshape(B * Q),
            weight=self.empty_weight,
        )

        # ── 박스 손실 (매칭된 쌍만) ─────────────────────────────────────────
        matched_pred_boxes = []
        matched_gt_boxes = []
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            matched_pred_boxes.append(pred_boxes[b][pred_idx])
            matched_gt_boxes.append(targets[b]["boxes"][gt_idx].to(pred_boxes.device))

        if len(matched_pred_boxes) == 0:
            loss_bbox = pred_boxes.sum() * 0
            loss_giou = pred_boxes.sum() * 0
        else:
            mp = torch.cat(matched_pred_boxes, dim=0)
            mg = torch.cat(matched_gt_boxes, dim=0)
            loss_bbox = F.l1_loss(mp, mg, reduction='mean')
            loss_giou = (1 - generalized_iou(mp, mg).diag()).mean()

        total = (self.lambda_cls * loss_cls
                 + self.lambda_bbox * loss_bbox
                 + self.lambda_giou * loss_giou)

        return total, {
            "loss_cls": loss_cls.item(),
            "loss_bbox": loss_bbox.item(),
            "loss_giou": loss_giou.item(),
        }


# ─── mAP 계산 ────────────────────────────────────────────────────────────────

def calculate_IoU(box1, box2):
    """[cx,cy,w,h] → IoU"""
    b1x1, b1y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    b1x2, b1y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    b2x1, b2y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    b2x2, b2y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    ix1, iy1 = max(b1x1, b2x1), max(b1y1, b2y1)
    ix2, iy2 = min(b1x2, b2x2), min(b1y2, b2y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (b1x2 - b1x1) * (b1y2 - b1y1) + (b2x2 - b2x1) * (b2y2 - b2y1) - inter
    return inter / (union + 1e-6)


def get_detr_bboxes(loader, model, threshold=0.5, device="cuda", num_classes=20):
    """
    모델 추론 후 NMS 없이 confidence threshold 기반으로 bbox 추출.
    반환: all_pred_boxes, all_true_boxes (YOLO mAP 형식)
    [image_idx, class_id, score, cx, cy, w, h]
    """
    model.eval()
    all_pred_boxes = []
    all_true_boxes = []
    image_idx = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            pred_logits, pred_boxes = model(images)

            prob = pred_logits.softmax(-1)                      # (B, Q, C+1)
            scores, labels = prob[..., :num_classes].max(-1)   # (B, Q)

            B = images.shape[0]
            for b in range(B):
                # 예측 박스
                for q in range(pred_boxes.shape[1]):
                    if scores[b, q].item() > threshold:
                        all_pred_boxes.append([
                            image_idx,
                            labels[b, q].item(),
                            scores[b, q].item(),
                            *pred_boxes[b, q].tolist(),
                        ])

                # GT 박스
                n = targets["num_objects"][b].item()
                for i in range(n):
                    all_true_boxes.append([
                        image_idx,
                        targets["labels"][b, i].item(),
                        1.0,
                        *targets["boxes"][b, i].tolist(),
                    ])

                image_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def mAP(predict_boxes, gt_boxes, iou_threshold=0.5, num_classes=20):
    average_precisions = []
    epsilon = 1e-6

    for cls in range(num_classes):
        dets = [d for d in predict_boxes if d[1] == cls]
        gts = [g for g in gt_boxes if g[1] == cls]

        if len(gts) == 0:
            continue

        # 이미지당 GT 개수
        img_gt_count = {}
        for g in gts:
            img_gt_count[g[0]] = img_gt_count.get(g[0], 0) + 1
        img_gt_used = {k: torch.zeros(v) for k, v in img_gt_count.items()}

        dets.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(dets))
        FP = torch.zeros(len(dets))

        for d_idx, det in enumerate(dets):
            img_gts = [g for g in gts if g[0] == det[0]]
            best_iou, best_gt_idx = 0, -1
            for g_idx, gt in enumerate(img_gts):
                iou = calculate_IoU(det[3:7], gt[3:7])
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, g_idx

            if best_iou > iou_threshold and best_gt_idx >= 0:
                if img_gt_used[det[0]][best_gt_idx] == 0:
                    TP[d_idx] = 1
                    img_gt_used[det[0]][best_gt_idx] = 1
                else:
                    FP[d_idx] = 1
            else:
                FP[d_idx] = 1

        TP_cs = torch.cumsum(TP, dim=0)
        FP_cs = torch.cumsum(FP, dim=0)
        recalls = TP_cs / (len(gts) + epsilon)
        precisions = TP_cs / (TP_cs + FP_cs + epsilon)
        precisions = torch.cat([torch.tensor([1.0]), precisions])
        recalls = torch.cat([torch.tensor([0.0]), recalls])
        average_precisions.append(torch.trapz(precisions, recalls).item())

    return sum(average_precisions) / max(len(average_precisions), 1)
