import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.ops import nms

from classify import MultiTaskModel, CarDataset, collate_fn, DETECTION_CLASSES


# Reverse mapping: class_id -> class_name
CLASS_NAMES = {v + 1: k for k, v in DETECTION_CLASSES.items()}


def apply_nms(predictions, iou_threshold=0.3, score_threshold=0.5):
    """Apply per-class NMS filtering"""
    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']

    mask = scores >= score_threshold
    boxes, labels, scores = boxes[mask], labels[mask], scores[mask]

    if len(boxes) == 0:
        return {'boxes': torch.zeros((0, 4)), 'labels': torch.zeros(0, dtype=torch.long), 'scores': torch.zeros(0)}

    final_boxes, final_labels, final_scores = [], [], []

    for label in labels.unique():
        mask = labels == label
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        final_boxes.append(boxes[mask][keep])
        final_labels.append(labels[mask][keep])
        final_scores.append(scores[mask][keep])

    return {
        'boxes': torch.cat(final_boxes) if final_boxes else torch.zeros((0, 4)),
        'labels': torch.cat(final_labels) if final_labels else torch.zeros(0, dtype=torch.long),
        'scores': torch.cat(final_scores) if final_scores else torch.zeros(0)
    }


class DetectionVisualizer:
    """Draw bounding boxes on images"""

    def __init__(self, class_names):
        self.class_names = class_names
        np.random.seed(42)
        self.colors = {
            class_id: tuple(map(int, np.random.randint(0, 255, 3)))
            for class_id in class_names.keys()
        }

    def draw_boxes(self, image, boxes, labels, scores=None):
        """Draw boxes on image"""
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image.copy()

        if len(img_np.shape) == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        for i in range(len(boxes)):
            box = boxes[i].cpu().numpy() if torch.is_tensor(boxes[i]) else boxes[i]
            label = int(labels[i].cpu().item() if torch.is_tensor(labels[i]) else labels[i])
            color = self.colors.get(label, (255, 255, 255))
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)

            text = self.class_names.get(label, f"Class {label}")
            if scores is not None:
                score = scores[i].cpu().item() if torch.is_tensor(scores[i]) else scores[i]
                text = f"{text}: {score:.2f}"

            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_np, (x1, y1 - text_h - baseline - 5), (x1 + text_w, y1), color, -1)
            cv2.putText(img_np, text, (x1, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

    def _tensor_to_pil(self, tensor):
        """Convert normalized tensor to PIL"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
        tensor = torch.clamp(tensor * std + mean, 0, 1)
        return Image.fromarray((tensor * 255).byte().permute(1, 2, 0).cpu().numpy())

    def compare_pred_and_gt(self, image, predictions, targets, save_path=None):
        """Side-by-side comparison"""
        if torch.is_tensor(image):
            image = self._tensor_to_pil(image)

        pred_img = self.draw_boxes(image.copy(), predictions['boxes'], predictions['labels'], predictions.get('scores'))
        gt_img = self.draw_boxes(image.copy(), targets['boxes'], targets['labels'])

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(gt_img)
        axes[0].set_title('Ground Truth', fontsize=16)
        axes[0].axis('off')
        axes[1].imshow(pred_img)
        axes[1].set_title(f'Predictions ({len(predictions["boxes"])} objects)', fontsize=16)
        axes[1].axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        plt.close()


def visualize_batch(model, dataloader, num_images=10, output_dir='detection_results',
                   score_threshold=0.5, nms_threshold=0.3):
    """Visualize predictions on batch"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    visualizer = DetectionVisualizer(CLASS_NAMES)
    model.eval()
    device = next(model.parameters()).device

    count = 0
    total_raw = 0
    total_filtered = 0

    with torch.no_grad():
        for images, labels_dict, bbox_targets in dataloader:
            images_list = [img.to(device) for img in images]
            _, predictions = model(images_list, targets=None)

            for i in range(len(images)):
                if count >= num_images:
                    print(f"\n{'='*70}")
                    print(f"Summary: {count} images")
                    print(f"  Avg raw: {total_raw/count:.1f}")
                    print(f"  Avg filtered: {total_filtered/count:.1f}")
                    print(f"  Reduction: {(1-total_filtered/max(total_raw,1))*100:.1f}%")
                    print(f"✓ Saved to {output_dir}/")
                    print(f"{'='*70}")
                    return

                # Ground truth
                if len(bbox_targets[i]) > 0:
                    h, w = images[i].shape[1:]
                    boxes_gt = bbox_targets[i][:, 1:5].clone()
                    labels_gt = bbox_targets[i][:, 0].long()

                    boxes_gt[:, [0, 2]] *= w
                    boxes_gt[:, [1, 3]] *= h

                    boxes_corner = torch.zeros_like(boxes_gt)
                    boxes_corner[:, 0] = boxes_gt[:, 0] - boxes_gt[:, 2] / 2
                    boxes_corner[:, 1] = boxes_gt[:, 1] - boxes_gt[:, 3] / 2
                    boxes_corner[:, 2] = boxes_gt[:, 0] + boxes_gt[:, 2] / 2
                    boxes_corner[:, 3] = boxes_gt[:, 1] + boxes_gt[:, 3] / 2

                    targets_vis = {'boxes': boxes_corner, 'labels': labels_gt + 1}
                else:
                    targets_vis = {'boxes': torch.zeros((0, 4)), 'labels': torch.zeros(0, dtype=torch.long)}

                # Predictions
                pred_vis = {
                    'boxes': predictions[i]['boxes'].cpu(),
                    'labels': predictions[i]['labels'].cpu(),
                    'scores': predictions[i]['scores'].cpu()
                }

                total_raw += len(pred_vis['boxes'])
                filtered = apply_nms(pred_vis, nms_threshold, score_threshold)
                total_filtered += len(filtered['boxes'])

                print(f"Image {count:03d}: {len(pred_vis['boxes'])} → {len(filtered['boxes'])} (GT: {len(targets_vis['boxes'])})")

                save_path = output_path / f"detection_{count:03d}.jpg"
                visualizer.compare_pred_and_gt(images[i], filtered, targets_vis, save_path=save_path)
                count += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num-images', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default='detection_results')
    parser.add_argument('--score-threshold', type=float, default=0.5)
    parser.add_argument('--nms-threshold', type=float, default=0.3)
    args = parser.parse_args()

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = MultiTaskModel.load_from_checkpoint(
        args.checkpoint,
        backbone_name="efficientnet_b0",
        num_detection_classes=12,
    )
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print("✓ Model loaded")

    # Dataset
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = CarDataset(
        image_dir="data/car-img-hr",
        json_file="data/ground_truth/project-6-at-2025-11-18-10-24-3493761b.json",
        transform=transform,
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    print(f"\nVisualizing {args.num_images} images...")
    print(f"  Score threshold: {args.score_threshold}")
    print(f"  NMS threshold: {args.nms_threshold}\n")

    visualize_batch(model, dataloader, num_images=args.num_images, 
                   output_dir=args.output_dir,
                   score_threshold=args.score_threshold,
                   nms_threshold=args.nms_threshold)
