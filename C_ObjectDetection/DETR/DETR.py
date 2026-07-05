import os
import sys
import math
import yaml

_project_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import random_split

from I_Model_Zoo.Models.Util.ModelBase import modelbase
from C_ObjectDetection.DETR.DataLoader import DETRDataLoader
from C_ObjectDetection.Util.Draw_Graph import Draw_Graph
from C_ObjectDetection.DETR.Trainer import train_model


class PositionalEncoding2D(nn.Module):
    """
    DETR 논문의 2D sine/cosine positional encoding.
    feature map (B, C, H, W) → 위치 임베딩 (B, C, H, W) 추가 반환.
    """
    def __init__(self, hidden_dim=256, temperature=10000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.hidden_dim, "hidden_dim과 채널 수가 일치해야 합니다"

        y_embed = torch.arange(H, dtype=torch.float32, device=x.device).unsqueeze(1).expand(H, W)
        x_embed = torch.arange(W, dtype=torch.float32, device=x.device).unsqueeze(0).expand(H, W)

        dim_t = torch.arange(C // 2, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (C // 2))

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t

        pos_x = torch.stack([pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()], dim=-1).flatten(-2)

        pos = torch.cat([pos_y, pos_x], dim=-1).permute(2, 0, 1)  # (C, H, W)
        return pos.unsqueeze(0).expand(B, -1, -1, -1)              # (B, C, H, W)


# ─── DETR 모델 ───────────────────────────────────────────────────────────────

class DETR(modelbase):
    """
    DEtection TRansformer (DETR).

    구조:
      Backbone (ResNet50) → projection Conv → Transformer (Encoder+Decoder) → FFN Head

    출력:
      pred_logits: (B, num_queries, num_classes+1)  마지막 = no-object
      pred_boxes:  (B, num_queries, 4)              [cx,cy,w,h] sigmoid 적용 (0~1)
    """

    def __init__(self, num_classes=20, num_queries=100, hidden_dim=256,
                 nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        self.transform_info = transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # ── Backbone ─────────────────────────────────────────────────────────
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # layer4까지만 사용 (stride=32, 2048 채널 출력)
        self.backbone = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )

        # backbone 파라미터 일부 고정 (layer2까지)
        for name, param in self.backbone.named_parameters():
            if "4." not in name and "5." not in name and "6." not in name and "7." not in name:
                param.requires_grad_(False)

        # ── Input Projection ─────────────────────────────────────────────────
        self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)

        # ── Positional Encoding ──────────────────────────────────────────────
        self.pos_enc = PositionalEncoding2D(hidden_dim)

        # ── Transformer ──────────────────────────────────────────────────────
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )

        # ── Object Queries ───────────────────────────────────────────────────
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # ── Prediction Heads ─────────────────────────────────────────────────
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 = no-object
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        B = x.shape[0]

        # 백본 피처 추출: (B, 2048, H/32, W/32)
        feat = self.backbone(x)

        # hidden_dim으로 프로젝션: (B, hidden_dim, H', W')
        feat = self.input_proj(feat)

        # 위치 인코딩 추가
        pos = self.pos_enc(feat)
        feat = feat + pos

        # Transformer 입력: (B, H'*W', hidden_dim)
        H, W = feat.shape[2], feat.shape[3]
        src = feat.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        # Object queries: (B, num_queries, hidden_dim)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # Transformer 통과
        hs = self.transformer(src, queries)  # (B, num_queries, hidden_dim)

        # 예측 헤드
        pred_logits = self.class_embed(hs)            # (B, Q, num_classes+1)
        pred_boxes = self.bbox_embed(hs).sigmoid()    # (B, Q, 4) in [0,1]

        return pred_logits, pred_boxes

    def return_transform_info(self):
        return self.transform_info


# ─── Pretrained DETR (Facebook Research) ─────────────────────────────────────

class PretrainedDETR(modelbase):
    """
    Facebook Research 공식 pretrained DETR (detr_resnet50).
    COCO 80클래스로 pretrained → 헤드만 교체하여 Fine-tuning.
    """
    def __init__(self, num_classes=20, num_queries=100):
        super().__init__()

        self.transform_info = transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 공식 pretrained 모델 로드 (COCO 91클래스)
        self.model = torch.hub.load(
            'facebookresearch/detr', 'detr_resnet50',
            pretrained=True, num_classes=91
        )

        hidden_dim = self.model.transformer.d_model  # 256

        # 헤드를 타겟 클래스 수로 교체 (기존 가중치 버리고 새로 학습)
        self.model.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.model.query_embed = nn.Embedding(num_queries, hidden_dim)

        # backbone은 고정, transformer + 헤드만 학습
        for param in self.model.backbone.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        out = self.model(x)
        return out['pred_logits'], out['pred_boxes']

    def return_transform_info(self):
        return self.transform_info


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.normpath(os.path.join(current_dir, '..', 'Util', 'config.yaml'))

    with open(yaml_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    NUM_CLASSES = config.get('num_class', 20)
    IMAGE_SIZE = config.get('DETR_IMAGE_SIZE', 800)
    NUM_QUERIES = 100
    BATCH_SIZE = config.get('DETR_BATCH_SIZE', 2)
    EPOCHS = config.get('epoch', 300)
    PATIENCE = config.get('patience', 50)

    USE_PRETRAINED = True  # False로 바꾸면 처음부터 학습

    if USE_PRETRAINED:
        print("Pretrained DETR (facebookresearch/detr_resnet50) 로드 중...")
        model = PretrainedDETR(num_classes=NUM_CLASSES, num_queries=NUM_QUERIES).to(device)
    else:
        model = DETR(
            num_classes=NUM_CLASSES,
            num_queries=NUM_QUERIES,
            hidden_dim=256,
            nheads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
        ).to(device)

    graph = Draw_Graph(model=model, save_path=config['save_path'], patience=PATIENCE)

    train_val_set = DETRDataLoader(
        csv_file=config['traincsvfile_path'],
        img_dir=config['IMG_DIR'],
        label_dir=config['LABEL_DIR'],
        image_size=IMAGE_SIZE,
        max_objects=NUM_QUERIES,
        transform='train',
    )

    test_set = DETRDataLoader(
        csv_file=config['testcsvfile_path'],
        img_dir=config['IMG_DIR'],
        label_dir=config['LABEL_DIR'],
        image_size=IMAGE_SIZE,
        max_objects=NUM_QUERIES,
        transform='test',
    )

    train_num = int(0.8 * len(train_val_set))
    val_num = len(train_val_set) - train_num
    train_set, val_set = random_split(train_val_set, [train_num, val_num])

    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images, dim=0)
        boxes = torch.stack([t["boxes"] for t in targets], dim=0)
        labels = torch.stack([t["labels"] for t in targets], dim=0)
        num_objects = torch.tensor([t["num_objects"] for t in targets])
        return images, {"boxes": boxes, "labels": labels, "num_objects": num_objects}

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    train_model(
        device=device,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        graph=graph,
        num_classes=NUM_CLASSES,
        epochs=EPOCHS,
        lr=1e-4,
        weight_decay=1e-4,
        patience=PATIENCE,
        graph_update_epoch=1,
    )


if __name__ == "__main__":
    main()
