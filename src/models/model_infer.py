"""
model_infer.py
charge a trained RF-DETR model and perform inference on extracted video frames,"""

import os
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from collections import Counter
import supervision as sv
from rfdetr import RFDETRSmall
import json

# Configuración de rutas y parámetros
top_dir = Path(__file__).parent.parent.parent
FRAMES_DIR = Path(__file__).parent.parent / "frames-output"
CHECKPOINT_PATH = top_dir / "models" / "RF-DETR" / "checkpoint_best_ema.pth"
TRAIN_JSON = top_dir / "data" / "Pieces Count.v1-dataset-basis-5-classes-24-11-2025.coco" / "train" / "_annotations.coco.json"
CONF_THRESHOLD = 0.4

with open(TRAIN_JSON, 'r', encoding='utf-8') as f:
    categories = json.load(f)["categories"]
    NUM_CLASSES = len(categories)
    CLASS_NAMES = [cat["name"] for cat in categories]

# Cargar modelo
model = RFDETRSmall(num_classes=NUM_CLASSES)
state = torch.load(CHECKPOINT_PATH, map_location="cpu")
if "model" in state:
    state = state["model"]
elif "ema" in state:
    state = state["ema"]
elif "state_dict" in state:
    state = state["state_dict"]

try:
    torch_model = model.model.model
except AttributeError:
    torch_model = model.model

torch_model.load_state_dict(state, strict=False)
model.optimize_for_inference()
model.names = CLASS_NAMES

# Procesar cada frame

def process_frame(frame_file):
    """
    Procesa un frame y retorna el resultado como string.
    """
    image_pil = Image.open(frame_file).convert("RGB")
    detections_model_raw = model.predict(image_pil, conf_threshold=CONF_THRESHOLD)
    adjusted_detection_class_ids = [int(cid) for cid in detections_model_raw.class_id]
    valid_indices = [i for i, cid in enumerate(adjusted_detection_class_ids) if 0 <= cid < NUM_CLASSES]
    detections_adjusted = sv.Detections(
        xyxy=detections_model_raw.xyxy[valid_indices],
        confidence=detections_model_raw.confidence[valid_indices],
        class_id=np.array([adjusted_detection_class_ids[i] for i in valid_indices], dtype=int),
    )
    detected_types = [CLASS_NAMES[int(class_id)] for class_id in detections_adjusted.class_id]
    conteo_tipos = Counter(detected_types)
    result = f"Frame: {Path(frame_file).name}\nConteo de piezas detectadas por tipo:\n"
    for tipo, cantidad in conteo_tipos.items():
        if tipo == "Alerones traseros":
            result += f"  {tipo}: 1\n"
        else:
            result += f"  {tipo}: {cantidad}\n"
    return result
