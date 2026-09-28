import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.transforms import functional as F

DEFAULT_IMG_SIZE = 384


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        p_left, p_top = (max_wh - w) // 2, (max_wh - h) // 2
        p_right, p_bottom = max_wh - w - p_left, max_wh - h - p_top
        return F.pad(
            image, [p_left, p_top, p_right, p_bottom], fill=255, padding_mode="constant"
        )


class ResNetClassifier:
    def __init__(self, model_path: str, classes_path: str, use_padding: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(classes_path):
            raise FileNotFoundError(f"Classes file not found: {classes_path}")

        with open(classes_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.model = models.resnet50(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, len(self.classes))

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        transform_ops = []
        if use_padding:
            transform_ops.append(SquarePad())
        transform_ops.extend(
            [
                transforms.Resize((DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.transform = transforms.Compose(transform_ops)

    def predict(self, image_path: str) -> str | None:
        try:
            img = Image.open(image_path).convert("RGB")
            tensor_img = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(tensor_img)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, pred_idx = torch.max(probabilities, 1)
            return self.classes[int(pred_idx.item())]
        except Exception:
            return None


def load_metadata_map(data_dir: str) -> Dict[str, str]:
    """Reads pdffigures2 metadata JSONs to pick out figures already tagged as tables."""
    meta_map = {}
    if not os.path.exists(data_dir):
        return meta_map
    for jf in Path(data_dir).glob("*.json"):
        try:
            with open(jf) as f:
                data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        if "renderURL" in entry and entry.get("figType") == "Table":
                            fname = Path(entry["renderURL"]).name
                            meta_map[fname] = "tables"
        except Exception:
            pass
    return meta_map


def classify_and_filter(
    input_dir: str,
    output_dir: str,
    model_path: str,
    classes_path: str,
    use_padding: bool = False,
):
    """Classifies extracted figure images into category subfolders of output_dir."""
    perf_log = []

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    input_images_dir = os.path.join(input_dir, "images")
    input_data_dir = os.path.join(input_dir, "data")

    if not os.path.exists(input_images_dir):
        raise FileNotFoundError(f"Input folder missing: {input_images_dir}")

    classifier = ResNetClassifier(model_path, classes_path, use_padding)
    meta_map = load_metadata_map(input_data_dir)

    images = [
        f for f in os.listdir(input_images_dir) if f.lower().endswith((".png", ".jpg"))
    ]

    for filename in images:
        t_item_start = time.time()

        src_path = os.path.join(input_images_dir, filename)
        label = meta_map.get(filename)
        source_method = "metadata"

        if not label:
            label = classifier.predict(src_path)
            source_method = "ai_inference"

        if label:
            clean_label = label.replace(" ", "_").lower()
            target_folder = os.path.join(output_dir, clean_label)
            os.makedirs(target_folder, exist_ok=True)
            shutil.copy2(src_path, os.path.join(target_folder, filename))

            perf_log.append(
                {
                    "filename": filename,
                    "label": clean_label,
                    "method": source_method,
                    "time_seconds": time.time() - t_item_start,
                }
            )

    return perf_log
