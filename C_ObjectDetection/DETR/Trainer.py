import os
import sys

_project_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tqdm import tqdm
import torch
import torch.optim as optim
from C_ObjectDetection.DETR.Util import DETRLoss, get_detr_bboxes, mAP


def train_model(device, model, train_loader, val_loader, test_loader, graph,
                num_classes, epochs=300, lr=1e-4, weight_decay=1e-4,
                patience=50, graph_update_epoch=1):

    # DETR 논문 권장: backbone은 작은 lr, transformer는 큰 lr
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
         "lr": lr * 0.1},
    ]
    optimizer = optim.AdamW(param_dicts, lr=lr, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

    loss_fn = DETRLoss(num_classes=num_classes).to(device)
    scaler = torch.amp.GradScaler('cuda')

    best_val_acc = 0
    patience_count = 0

    pbar = tqdm(range(epochs), desc="Epoch Progress")

    for epoch in pbar:
        model.train()
        running_loss = 0.0
        running_cls = 0.0
        running_bbox = 0.0
        running_giou = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)

            # targets의 각 텐서를 device로 이동
            targets_dev = {
                "boxes": targets["boxes"].to(device),
                "labels": targets["labels"].to(device),
                "num_objects": targets["num_objects"],
            }
            # list of dict 형태로 변환 (배치별 분리)
            B = images.shape[0]
            targets_list = [
                {
                    "boxes": targets_dev["boxes"][b],
                    "labels": targets_dev["labels"][b],
                    "num_objects": targets_dev["num_objects"][b].item(),
                }
                for b in range(B)
            ]

            with torch.amp.autocast('cuda'):
                pred_logits, pred_boxes = model(images)
                loss, loss_dict = loss_fn(pred_logits, pred_boxes, targets_list)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # gradient clipping (DETR 학습 안정화)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_cls += loss_dict["loss_cls"]
            running_bbox += loss_dict["loss_bbox"]
            running_giou += loss_dict["loss_giou"]

        lr_scheduler.step()

        n_batches = max(len(train_loader), 1)
        train_loss = running_loss / n_batches
        avg_cls = running_cls / n_batches
        avg_bbox = running_bbox / n_batches
        avg_giou = running_giou / n_batches

        # Validation Loss 계산
        if epoch % graph_update_epoch == 0:
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for images_v, targets_v in val_loader:
                    images_v = images_v.to(device)
                    B_v = images_v.shape[0]
                    targets_v_list = [
                        {
                            "boxes": targets_v["boxes"][b].to(device),
                            "labels": targets_v["labels"][b].to(device),
                            "num_objects": targets_v["num_objects"][b].item(),
                        }
                        for b in range(B_v)
                    ]
                    with torch.amp.autocast('cuda'):
                        pred_logits_v, pred_boxes_v = model(images_v)
                        val_loss_batch, _ = loss_fn(pred_logits_v, pred_boxes_v, targets_v_list)
                    val_running_loss += val_loss_batch.item()
            val_loss = val_running_loss / max(len(val_loader), 1)
            model.train()

            # Train mAP 계산
            train_pred_boxes, train_gt_boxes = get_detr_bboxes(
                train_loader, model, threshold=0.1, device=device, num_classes=num_classes
            )
            train_mAP = mAP(train_pred_boxes, train_gt_boxes, iou_threshold=0.1, num_classes=num_classes)

            # Val mAP 계산
            val_pred_boxes, val_gt_boxes = get_detr_bboxes(
                val_loader, model, threshold=0.1, device=device, num_classes=num_classes
            )
            val_mAP = mAP(val_pred_boxes, val_gt_boxes, iou_threshold=0.1, num_classes=num_classes)

            pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'Train mAP': f'{train_mAP:.4f}',
                'Val mAP': f'{val_mAP:.4f}',
            })

            graph.update_graph(
                train_acc=train_mAP,
                train_loss=train_loss,
                val_acc=val_mAP,
                val_loss=val_loss,
                epoch=epoch,
                patience_count=patience_count,
            )

            if val_mAP > best_val_acc:
                best_val_acc = val_mAP
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break
        else:
            pbar.set_postfix({'Loss': f'{train_loss:.4f}'})

    graph.save_model()
