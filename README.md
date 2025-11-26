## Data & Model Access

**Note:** Large files such as the trained model weights, full dataset, and other heavy resources are not included in this repository.

To request access to the dataset and training files, please send an email to juanpagonzalez457@gmail.com with the subject:

`Solicitud Base de Datos y Entrenamiento`

# YamaLens

YamaLens is a modular Python application for automatic piece counting in industrial videos using AI. It extracts frames from a video, runs inference with a trained model, and displays results in a modern GUI.

## Features
- **Frame Extraction:** Extracts frames from video at user-defined intervals.
- **Model Inference:** Uses a custom RF-DETR model to predict piece types and counts for each frame.
- **Modern GUI:** PyQt5 interface with welcome screen, styled buttons, input fields, and navigation arrows for results.
- **Results Viewer:** Displays each frame and its prediction, allowing navigation between results.

## Project Structure
```
YamaLens/
├── data/
│   └── Pieces Count.v1-dataset-basis-5-classes-24-11-2025.coco/
│       ├── train/
│       ├── test/
│       └── valid/
├── models/
│   └── RF-DETR/
│       ├── checkpoint_best_ema.pth
│       ├── checkpoint_best_regular.pth
│       ├── rf-detr-small.pth
│       └── ...
├── notebooks/
│   └── RF-DETR.ipynb
├── reports/
├── src/
│   ├── frame-extraction/
│   │   └── frame_extractor.py
│   ├── models/
│   │   └── model_infer.py
│   ├── gui/
│   │   └── gui_app.py
│   └── frames-output/
├── requirements.txt
└── README.md
```

## How It Works
1. **Select a video** in the GUI.
2. **Set the frame interval** (seconds between frames).
3. **Run prediction:**
	 - Frames are extracted and saved.
	 - Each frame is processed by the model.
	 - Results (image + piece count/type) are shown in the GUI.
4. **Navigate results** using arrows; return to welcome screen to process another video.

## Installation
1. Clone the repository:
	```sh
	git clone <repo-url>
	cd YamaLens
	```
2. Create and activate a Python virtual environment:
	```sh
	python -m venv .venv
	.venv\Scripts\activate  # Windows
	source .venv/bin/activate  # Linux/Mac
	```
3. Install dependencies:
	```sh
	pip install -r requirements.txt
	```

## Usage
Run the GUI application:
```sh
python src/gui/gui_app.py
```

## Requirements
- Python 3.8+
- PyQt5
- OpenCV
- Torch (for RF-DETR model)

## Authors
- JuanGonzalez47
- Project collaborators (see `reports/integrantes.txt.txt`)

## License
MIT
