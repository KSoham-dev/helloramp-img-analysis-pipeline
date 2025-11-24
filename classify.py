import os
import re
import json
import torch
import torch.nn as nn
import pytorch_lightning as pl
from PIL import Image
from pathlib import Path
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset, DataLoader


# Task definitions
scene_tasks = {
    "car_present": 2,
    "car_cropped": 2,
    "distance_to_camera": 3,
    "environment": 2,
    "shoot_category": 3,
}
visibility_tasks = {
    "door_state": 3,
    "license_plate_visible": 2,
    "tyre_visible": 2,
    "tinted_windows": 2,
}
image_quality_tasks = {
    "photo_realism": 2,
    "reflection_level": 3,
    "image_exposure": 3,
    "dirt_on_car": 2,
    "dirt_on_tyre": 2,
}


class CarDataset(Dataset):
    """Custom Dataset for car images with multiple labels"""

    def __init__(self, image_dir, json_file, transform=None):
        """
        Args:
            image_dir: Directory with car images
            json_file: Path to Label Studio JSON file
            transform: Optional transform to be applied on images
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_paths = list(self.image_dir.glob("*.jpg"))
        self.classes = {
            "car_present": ["true", "false"],
            "car_cropped": ["true", "false"],
            "distance_to_camera": ["far", "normal", "close"],
            "environment": ["indoor", "outdoor"],
            "shoot_category": ["exterior", "closeup", "interior"],
            "photo_realism": ["real", "fake"],
            "reflection_level": ["low", "normal", "high"],
            "image_exposure": ["underexposed", "normal", "overexposed"],
            "license_plate_visible": ["true", "false"],
            "tyre_visible": ["true", "false"],
            "tinted_windows": ["true", "false"],
            "door_state": ["open_door", "closed_door", "boot_open"],
            "dirt_on_car": ["true", "false"],
            "dirt_on_tyre": ["true", "false"],
        }

        with open(json_file, "r") as f:
            data = json.load(f)

        # Create mapping from filename to labels
        self.label_dict = {}
        for annotation in data:
            if annotation["annotations"]:
                label_studio_filename = os.path.basename(annotation["file_upload"])
                actual_filename = re.sub(r"^[a-f0-9]+-", "", label_studio_filename)

                # Extract labels for all tasks
                fields = (
                    list(scene_tasks.keys())
                    + list(visibility_tasks.keys())
                    + list(image_quality_tasks.keys())
                )
                labels = {
                    field: next(
                        (
                            r["value"]["choices"][0]
                            for r in annotation["annotations"][0]["result"]
                            if r.get("from_name") == field
                        ),
                        None,
                    )
                    for field in fields
                }

                self.label_dict[actual_filename] = {
                    field: (
                        self.classes[field].index(labels[field])
                        if labels[field] is not None
                        else -100  # Missing labels automatically ignored by CrossEntropyLoss
                    )
                    for field in fields
                }

        self.image_paths = [
            p for p in self.image_paths if os.path.basename(p) in self.label_dict
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Returns a single sample"""
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Get labels for this image
        filename = os.path.basename(img_path)
        labels = self.label_dict[filename]

        return image, labels


class MultiTaskHead(nn.Module):
    def __init__(self, in_features, task_output_sizes):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.2)
        )
        self.task_heads = nn.ModuleDict(
            {
                name: nn.Linear(512, num_classes)
                for name, num_classes in task_output_sizes.items()
            }
        )

    def forward(self, x):
        features = self.shared(x)
        return {name: head(features) for name, head in self.task_heads.items()}


class MultiHeadEfficientNet(pl.LightningModule):
    def __init__(
        self,
        model_name="efficientnet-b5",
        freeze_backbone=True,
        lr=0.001,
        dataset=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["dataset"])
        self.lr = lr
        self.dataset = dataset

        # Load pretrained backbone
        self.backbone = EfficientNet.from_pretrained(model_name)
        in_features = self.backbone._fc.in_features

        # Remove original classifier
        self.backbone._fc = nn.Identity()

        # Freeze backbone if needed
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Define the three grouped heads
        self.scene = MultiTaskHead(in_features, scene_tasks)
        self.visibility = MultiTaskHead(in_features, visibility_tasks)
        self.image_quality = MultiTaskHead(in_features, image_quality_tasks)

        # Define loss functions (ignore_index=-100 by default)
        self.criterion_dict = {}
        for task in list(scene_tasks.keys()) + list(visibility_tasks.keys()) + list(image_quality_tasks.keys()):
            self.criterion_dict[task] = nn.CrossEntropyLoss()

    def forward(self, x):
        features = self.backbone(x)
        return {
            "scene": self.scene(features),
            "visibility": self.visibility(features),
            "image_quality": self.image_quality(features),
        }

    def training_step(self, batch, batch_idx):
        images, labels_dict = batch

        # Forward pass
        outputs = self(images)

        total_loss = 0
        valid_tasks = []
        skipped_tasks = []
        task_losses = {}

        # Loop over heads
        for head_name in ["scene", "visibility", "image_quality"]:
            out = outputs[head_name]
            for task_name, task_output in out.items():
                task_labels = labels_dict[task_name]

                # Check if there are any valid labels in this batch for this task
                valid_mask = (task_labels != -100)
                num_valid = valid_mask.sum().item()

                if num_valid > 0:  # Only compute loss if valid labels exist
                    loss = self.criterion_dict[task_name](task_output, task_labels)

                    # Additional safety check for nan
                    if not torch.isnan(loss):
                        total_loss += loss
                        task_losses[f"train/{task_name}_loss"] = loss.item()
                        valid_tasks.append(f"{task_name}({num_valid}/{len(task_labels)})")
                    else:
                        skipped_tasks.append(f"{task_name}(NaN)")
                else:
                    skipped_tasks.append(f"{task_name}(0/{len(task_labels)})")

        # Log metrics
        if len(valid_tasks) > 0:
            self.log("train/total_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train/valid_task_count", len(valid_tasks), prog_bar=True)
            
            # Log individual task losses
            self.log_dict(task_losses, on_step=False, on_epoch=True)

            # Print detailed info every 10 batches
            if batch_idx % 10 == 0:
                print(f"\nBatch [{batch_idx}], Loss: {total_loss.item():.4f}")
                print(f"  Valid Tasks ({len(valid_tasks)}): {', '.join(valid_tasks)}")
                if skipped_tasks:
                    print(f"  Skipped Tasks ({len(skipped_tasks)}): {', '.join(skipped_tasks)}")

            return total_loss
        else:
            print(f"\nBatch [{batch_idx}] SKIPPED - No valid labels")
            print(f"  All Tasks Skipped ({len(skipped_tasks)}): {', '.join(skipped_tasks)}")
            return None

    def validation_step(self, batch, batch_idx):
        images, labels_dict = batch

        # Forward pass
        outputs = self(images)

        total_loss = 0
        valid_count = 0
        task_losses = {}

        # Loop over heads
        for head_name in ["scene", "visibility", "image_quality"]:
            out = outputs[head_name]
            for task_name, task_output in out.items():
                task_labels = labels_dict[task_name]

                # Check if there are any valid labels
                valid_mask = (task_labels != -100)

                if valid_mask.sum() > 0:
                    loss = self.criterion_dict[task_name](task_output, task_labels)

                    if not torch.isnan(loss):
                        total_loss += loss
                        task_losses[f"val/{task_name}_loss"] = loss.item()
                        valid_count += 1

        # Log metrics
        if valid_count > 0:
            self.log("val/total_loss", total_loss, prog_bar=True, on_epoch=True)
            self.log("val/valid_task_count", valid_count)
            self.log_dict(task_losses, on_epoch=True)
            return total_loss

        return None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# Main execution
if __name__ == "__main__":
    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create dataset
    dataset = CarDataset(
        image_dir="data/car-img-hr",
        json_file="data/ground_truth/project-6-at-2025-11-18-10-24-3493761b.json",
        transform=transform,
    )

    # Split dataset (example: 80/20 train/val split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize model
    model = MultiHeadEfficientNet(
        model_name="efficientnet-b5", 
        freeze_backbone=True, 
        lr=0.001,
        dataset=dataset
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",  # Automatically uses GPU if available
        devices=1,
        log_every_n_steps=10,
        enable_checkpointing=True,
        default_root_dir="./lightning_logs",
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Visualize results
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
