import os
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
import timm


# Task definitions
SCENE_TASKS = {
    "car_present": 2, "car_cropped": 2, "distance_to_camera": 3,
    "environment": 2, "shoot_category": 3,
}

VISIBILITY_TASKS = {
    "door_state": 3, "license_plate_visible": 2,
    "tyre_visible": 2, "tinted_windows": 2,
}

QUALITY_TASKS = {
    "photo_realism": 2, "reflection_level": 3, "image_exposure": 3,
    "dirt_on_car": 2, "dirt_on_tyre": 2,
}

DETECTION_CLASSES = {
    "door": 0, "engine": 1, "front_seat": 2, "headlight": 3,
    "license_plate": 4, "rear_seat": 5, "roof": 6, "side_mirror": 7,
    "steering_wheel": 8, "tail_light": 9, "wheel": 10, "windshield": 11,
}

CLASSIFICATION_CLASSES = {
    "car_present": ["true", "false"], "car_cropped": ["true", "false"],
    "distance_to_camera": ["far", "normal", "close"],
    "environment": ["indoor", "outdoor"],
    "shoot_category": ["exterior", "closeup", "interior"],
    "photo_realism": ["real", "fake"],
    "reflection_level": ["low", "normal", "high"],
    "image_exposure": ["underexposed", "normal", "overexposed"],
    "license_plate_visible": ["true", "false"], "tyre_visible": ["true", "false"],
    "tinted_windows": ["true", "false"],
    "door_state": ["open_door", "closed_door", "boot_open"],
    "dirt_on_car": ["true", "false"], "dirt_on_tyre": ["true", "false"],
}


class CarDataset(Dataset):
    """Dataset with classification labels and bounding boxes"""

    def __init__(self, image_dir, json_file, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_paths = list(self.image_dir.glob("*.jpg"))

        # Load annotations
        with open(json_file, 'rb') as f:
            raw = f.read().replace(b'\r\n', b'\n').replace(b'\r', b'\n')
        data = json.loads(raw.decode('utf-8', 'ignore'))

        self.label_dict = {}
        self.bbox_dict = {}

        for annotation in data:
            if not annotation["annotations"]:
                continue

            # Extract filename
            label_studio_filename = os.path.basename(annotation["file_upload"])
            actual_filename = re.sub(r"^[a-f0-9]+-", "", label_studio_filename)

            # Parse classification labels
            fields = list(SCENE_TASKS.keys()) + list(VISIBILITY_TASKS.keys()) + list(QUALITY_TASKS.keys())
            labels = {
                field: next(
                    (r["value"]["choices"][0] for r in annotation["annotations"][0]["result"]
                     if r.get("from_name") == field), None
                )
                for field in fields
            }

            self.label_dict[actual_filename] = {
                field: CLASSIFICATION_CLASSES[field].index(labels[field]) if labels[field] else -100
                for field in fields
            }

            # Parse bounding boxes
            bboxes = []
            for r in annotation["annotations"][0]["result"]:
                if r.get("type") == "rectanglelabels":
                    bbox = r["value"]
                    class_name = bbox["rectanglelabels"][0]

                    if class_name in DETECTION_CLASSES:
                        x_center = (bbox["x"] + bbox["width"] / 2) / 100.0
                        y_center = (bbox["y"] + bbox["height"] / 2) / 100.0
                        width = bbox["width"] / 100.0
                        height = bbox["height"] / 100.0

                        bboxes.append({
                            "class": DETECTION_CLASSES[class_name],
                            "bbox": [x_center, y_center, width, height]
                        })

            self.bbox_dict[actual_filename] = bboxes

        # Keep only images with labels
        self.image_paths = [p for p in self.image_paths if p.name in self.label_dict]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        filename = img_path.name
        labels = self.label_dict[filename]

        # Convert bboxes to tensor [num_boxes, 5] = [class, x, y, w, h]
        bboxes = self.bbox_dict.get(filename, [])
        if len(bboxes) > 0:
            bbox_tensor = torch.zeros((len(bboxes), 5))
            for i, bbox in enumerate(bboxes):
                bbox_tensor[i, 0] = bbox["class"]
                bbox_tensor[i, 1:5] = torch.tensor(bbox["bbox"])
        else:
            bbox_tensor = torch.zeros((0, 5))

        return image, labels, bbox_tensor


def collate_fn(batch):
    """Custom collate for batching"""
    images, labels_dict, bboxes = zip(*batch)
    images = torch.stack(images)

    combined_labels = {}
    for key in labels_dict[0].keys():
        combined_labels[key] = torch.tensor([d[key] for d in labels_dict])

    return images, combined_labels, bboxes


class BackboneWithFPN(nn.Module):
    """Feature Pyramid Network backbone"""

    def __init__(self, backbone_name="efficientnet_b0"):
        super().__init__()

        self.body = timm.create_model(backbone_name, pretrained=True, features_only=True)

        # Use last 4 stages or all if fewer
        all_channels = [info['num_chs'] for info in self.body.feature_info]
        total_stages = len(all_channels)

        if total_stages <= 4:
            self.stage_indices = list(range(total_stages))
            self.in_channels_list = all_channels
        else:
            self.stage_indices = list(range(total_stages - 4, total_stages))
            self.in_channels_list = all_channels[-4:]

        self.num_stages = len(self.in_channels_list)

        # FPN layers
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, 256, 1) for ch in self.in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(256, 256, 3, padding=1) for _ in range(self.num_stages)
        ])

        self.out_channels = 256
        self.out_channels_list = [256] * self.num_stages

    def forward(self, x):
        all_features = self.body(x)
        features = [all_features[i] for i in self.stage_indices]

        # Build FPN
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]

        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[i], size=laterals[i-1].shape[-2:], mode='nearest')
            laterals[i-1] = laterals[i-1] + upsampled

        return {str(i): self.fpn_convs[i](laterals[i]) for i in range(len(laterals))}


class MultiTaskHead(nn.Module):
    """Multi-task classification head"""

    def __init__(self, in_features, task_output_sizes):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_features, 512),
            nn.ReLU(), nn.Dropout(0.2)
        )
        self.task_heads = nn.ModuleDict({
            name: nn.Linear(512, num_classes)
            for name, num_classes in task_output_sizes.items()
        })

    def forward(self, x):
        features = self.shared(x)
        return {name: head(features) for name, head in self.task_heads.items()}


class MultiTaskModel(pl.LightningModule):
    """Joint classification and detection model"""

    def __init__(self, backbone_name="efficientnet_b0", num_detection_classes=12,
                 lr=0.001, freeze_backbone=False):
        super().__init__()
        self.save_hyperparameters()

        backbone = BackboneWithFPN(backbone_name)

        # Detection head (Faster R-CNN)
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),) * backbone.num_stages,
            aspect_ratios=((0.5, 1.0, 2.0),) * backbone.num_stages
        )

        roi_pooler = MultiScaleRoIAlign(
            featmap_names=[str(i) for i in range(backbone.num_stages)],
            output_size=7, sampling_ratio=2
        )

        self.detector = FasterRCNN(
            backbone, num_classes=num_detection_classes + 1,
            rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler
        )

        if freeze_backbone:
            for param in self.detector.backbone.parameters():
                param.requires_grad = False

        # Classification heads
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.scene = MultiTaskHead(backbone.out_channels, SCENE_TASKS)
        self.visibility = MultiTaskHead(backbone.out_channels, VISIBILITY_TASKS)
        self.quality = MultiTaskHead(backbone.out_channels, QUALITY_TASKS)

        # Loss functions
        self.cls_losses = {
            task: nn.CrossEntropyLoss(ignore_index=-100)
            for task in list(SCENE_TASKS.keys()) + list(VISIBILITY_TASKS.keys()) + list(QUALITY_TASKS.keys())
        }

    def forward(self, images, targets=None):
        images_list = [img for img in images] if isinstance(images, torch.Tensor) else images
        image_tensor = torch.stack(images_list)

        # Get features
        backbone_features = self.detector.backbone(image_tensor)
        last_feature = list(backbone_features.values())[-1]

        # Classification
        pooled = torch.flatten(self.global_pool(last_feature), 1)
        cls_outputs = {
            "scene": self.scene(pooled),
            "visibility": self.visibility(pooled),
            "quality": self.quality(pooled),
        }

        # Detection
        if self.training and targets:
            det_losses = self.detector(images_list, targets)
            return cls_outputs, det_losses
        else:
            det_preds = self.detector(images_list)
            return cls_outputs, det_preds

    def training_step(self, batch, batch_idx):
        images, labels_dict, bbox_targets = batch

        # Prepare detection targets
        images_list = [img for img in images]
        targets = []
        for i, img in enumerate(images_list):
            h, w = img.shape[1:]

            if len(bbox_targets[i]) > 0:
                boxes = bbox_targets[i][:, 1:5].clone()
                labels = bbox_targets[i][:, 0].long()

                # Denormalize and convert to corner format
                boxes[:, [0, 2]] *= w
                boxes[:, [1, 3]] *= h

                boxes_corner = torch.zeros_like(boxes)
                boxes_corner[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
                boxes_corner[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
                boxes_corner[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
                boxes_corner[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

                targets.append({
                    'boxes': boxes_corner.to(self.device),
                    'labels': (labels + 1).to(self.device)
                })
            else:
                targets.append({
                    'boxes': torch.zeros((0, 4), device=self.device),
                    'labels': torch.zeros(0, dtype=torch.long, device=self.device)
                })

        # Forward pass
        cls_outputs, det_losses = self(images, targets)

        # Compute losses
        det_loss = sum(loss for loss in det_losses.values())

        cls_loss = 0
        for head_name in ["scene", "visibility", "quality"]:
            for task_name, task_output in cls_outputs[head_name].items():
                task_labels = labels_dict[task_name]
                if (task_labels != -100).sum() > 0:
                    loss = self.cls_losses[task_name](task_output, task_labels)
                    if not torch.isnan(loss):
                        cls_loss += loss

        total_loss = det_loss + cls_loss

        self.log("train/detection_loss", det_loss, prog_bar=True)
        self.log("train/classification_loss", cls_loss, prog_bar=True)
        self.log("train/total_loss", total_loss, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    # Data transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load dataset
    dataset = CarDataset(
        image_dir="data/car-img-hr",
        json_file="data/ground_truth/project-6-at-2025-11-18-10-24-3493761b.json",
        transform=transform,
    )

    # Train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, 
                            num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                          num_workers=2, collate_fn=collate_fn)

    # Model
    model = MultiTaskModel(backbone_name="efficientnet_b0", num_detection_classes=12, lr=0.001)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=4,
        gradient_clip_val=1.0,
        log_every_n_steps=5,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)