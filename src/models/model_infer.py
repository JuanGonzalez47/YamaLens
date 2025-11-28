from pathlib import Path
from PIL import Image
import torch
import numpy as np
from collections import Counter
import supervision as sv
from rfdetr import RFDETRSmall
import json
from ultralytics import YOLO

top_dir = Path(__file__).parent.parent.parent
FRAMES_DIR = Path(__file__).parent.parent / "frames-output"
CHECKPOINT_PATH = top_dir / "models" / "RF-DETR" / "checkpoint_best_ema.pth"
TRAIN_JSON = top_dir / "data" / "Pieces Count.v1-dataset-basis-5-classes-24-11-2025.coco" / "train" / "_annotations.coco.json"
CONF_THRESHOLD = 0.4

YOLO_WEIGHTS = top_dir / "models" / "YOLOv11" / "best.pt"
YOLO_LABEL_PATH = top_dir / "data" / "Pieces-Count-2" / "test" / "labels"

with open(TRAIN_JSON, 'r', encoding='utf-8') as f:
    categories = json.load(f)["categories"]
    NUM_CLASSES = len(categories)
    CLASS_NAMES = [cat["name"] for cat in categories]

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

# Cargar modelo YOLO solo si se usa
yolo_model = None
yolo_class_names = None
def load_yolo():
    global yolo_model, yolo_class_names
    if yolo_model is None:
        yolo_model = YOLO(str(YOLO_WEIGHTS))
        yolo_class_names = yolo_model.names

def process_frame_yolo(frame_file):
    """
    Procesa un frame con YOLO y retorna el resultado como string.
    """
    load_yolo()
    results = yolo_model(frame_file)
    detected_types = [yolo_class_names[int(cls)] for r in results for cls in r.boxes.cls.cpu().numpy().astype(int)]
    conteo_tipos = Counter(detected_types)
    result = f"Frame: {Path(frame_file).name}\nConteo de piezas detectadas por tipo (YOLO):\n"
    for tipo, cantidad in conteo_tipos.items():
        if tipo == "Alerones traseros":
            result += f"  {tipo}: 1\n"
        else:
            result += f"  {tipo}: {cantidad}\n"
    return result

def process_frame(frame_file, model_type='rfdetr'):
    """
    Procesa un frame usando el modelo seleccionado.
    """
    if model_type == 'yolo':
        return process_frame_yolo(frame_file)
    else:
        # RF-DETR por defecto
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
