import os
import re
import json
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

                # Extract car_present label
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
                        else -100 # Change this such that if the label is missing, loss set to zero (masked loss)
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


class MultiHeadEfficientNet(nn.Module):
    def __init__(self, model_name="efficientnet-b0", freeze_backbone=True):
        super().__init__()

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

    def forward(self, x):
        features = self.backbone(x)
        return {
            "scene": self.scene(features),
            "visibility": self.visibility(features),
            "image_quality": self.image_quality(features),
        }



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

# Create dataloader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Use 0 on Windows
    pin_memory=True,
)

model = MultiHeadEfficientNet(model_name="efficientnet-b5", freeze_backbone=True).to(
    device
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

criterion_dict = {}
for task in scene_tasks:
    criterion_dict[task] = nn.CrossEntropyLoss()
for task in visibility_tasks:
    criterion_dict[task] = nn.CrossEntropyLoss()
for task in image_quality_tasks:
    criterion_dict[task] = nn.CrossEntropyLoss()

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels_dict) in enumerate(train_loader):
        images = images.to(device)

        # Forward pass
        outputs = model(images)  # outputs: dict of dicts

        total_loss = 0
        valid_tasks = []
        skipped_tasks = []

        # Loop over heads
        for head_name in ["scene", "visibility", "image_quality"]:
            out = outputs[head_name]
            for task_name, task_output in out.items():
                task_labels = labels_dict[task_name].to(device)
                
                # Check if there are any valid labels in this batch for this task
                valid_mask = (task_labels != -100)
                num_valid = valid_mask.sum().item()
                
                if num_valid > 0:  # Only compute loss if valid labels exist
                    loss = criterion_dict[task_name](task_output, task_labels)
                    
                    # Additional safety check for nan
                    if not torch.isnan(loss):
                        total_loss += loss
                        valid_tasks.append(f"{task_name}({num_valid}/{len(task_labels)})")
                    else:
                        skipped_tasks.append(f"{task_name}(NaN)")
                else:
                    skipped_tasks.append(f"{task_name}(0/{len(task_labels)})")

        # Only backpropagate if we have at least one valid task loss
        if len(valid_tasks) > 0:
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Batch [{batch_idx}/{len(train_loader)}], "
                    f"Loss: {total_loss.item():.4f}"
                )
                print(f"  Valid Tasks ({len(valid_tasks)}): {', '.join(valid_tasks)}")
                if skipped_tasks:
                    print(f"  Skipped Tasks ({len(skipped_tasks)}): {', '.join(skipped_tasks)}")
        else:
            if batch_idx % 10 == 0:
                print(f"Batch [{batch_idx}] SKIPPED - No valid labels")
                print(f"  All Tasks Skipped ({len(skipped_tasks)}): {', '.join(skipped_tasks)}")

    avg_loss = running_loss / len(train_loader)
    print(f"\nEpoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}\n")

# print(f"Training complete. Output: {outputs}")