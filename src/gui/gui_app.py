"""
gui_app.py
user interface for YamaLens application using PyQt5."""

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QHBoxLayout, QStackedWidget, QLineEdit, QComboBox
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
import cv2

sys.path.append(str(Path(__file__).parent.parent / "frame_extraction"))
sys.path.append(str(Path(__file__).parent.parent / "models"))
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src" / "models"))
try:
    from model_infer import process_frame
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_infer", str(Path(__file__).parent.parent.parent / "src" / "models" / "model_infer.py"))
    model_infer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_infer)
    def process_frame(frame_file, model_type='rfdetr'):
        return model_infer.process_frame(frame_file, model_type)

# Importar la función de extracción de frames
try:
    from frame_extractor import extract_frames
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("frame_extractor", str(Path(__file__).parent.parent / "frame_extraction" / "frame_extractor.py"))
    frame_extractor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frame_extractor)
    extract_frames = frame_extractor.extract_frames

FRAMES_DIR = Path(__file__).parent / "frames-output"

class WelcomeWidget(QWidget):
    def __init__(self, start_callback):
        super().__init__()
        self.setStyleSheet("background-color: #000;")
        pal = self.palette()
        pal.setColor(self.backgroundRole(), Qt.black)
        self.setPalette(pal)
        self.setAutoFillBackground(True)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("Bienvenido a YamaLens")
        title.setFont(QFont("Arial", 32, QFont.Bold))
        title.setStyleSheet("color: white; margin-bottom: 20px;")
        layout.addWidget(title, alignment=Qt.AlignCenter)

        desc = QLabel("YamaLens es una herramienta para el conteo automático de piezas en videos industriales usando inteligencia artificial. Selecciona tu video y obtén predicciones precisas de cada tipo de pieza detectada.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: white; font-size: 16px; margin-bottom: 30px;")
        layout.addWidget(desc, alignment=Qt.AlignCenter)

        self.btn_select = QPushButton("Seleccionar video")
        self.btn_select.setStyleSheet("background-color: #ff1a1a; color: white; font-size: 18px; padding: 12px 32px; border-radius: 8px;")
        self.btn_select.setCursor(Qt.PointingHandCursor)
        self.btn_select.setMinimumWidth(250)
        self.btn_select.setMaximumWidth(350)
        self.btn_select.setFont(QFont("Arial", 16, QFont.Bold))
        self.btn_select.setToolTip("Selecciona un video para analizar")
        self.btn_select.enterEvent = lambda event: self.btn_select.setStyleSheet("background-color: #b20000; color: white; font-size: 18px; padding: 12px 32px; border-radius: 8px;")
        self.btn_select.leaveEvent = lambda event: self.btn_select.setStyleSheet("background-color: #ff1a1a; color: white; font-size: 18px; padding: 12px 32px; border-radius: 8px;")
        layout.addWidget(self.btn_select, alignment=Qt.AlignCenter)

        # Selector de modelo
        model_layout = QHBoxLayout()
        model_layout.setAlignment(Qt.AlignCenter)
        model_label = QLabel("Modelo de detección:")
        model_label.setStyleSheet("color: white; font-size: 16px;")
        self.model_selector = QComboBox()
        self.model_selector.addItems(["RF-DETR", "YOLOv11"])
        self.model_selector.setStyleSheet("background: white; color: black; font-size: 16px; border-radius: 6px; min-width: 100px;")
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_selector)
        layout.addLayout(model_layout)

        self.video_label = QLabel("")
        self.video_label.setStyleSheet("color: #ff1a1a; font-size: 16px; margin-bottom: 20px;")
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        interval_layout = QHBoxLayout()
        interval_layout.setAlignment(Qt.AlignCenter)
        interval_label = QLabel("Intervalo de frames (segundos):")
        interval_label.setStyleSheet("color: white; font-size: 16px;")
        self.interval_input = QLineEdit()
        self.interval_input.setPlaceholderText("1")
        self.interval_input.setStyleSheet("background: white; color: black; font-size: 16px; padding: 2px 4px; border-radius: 6px; min-width: 18px; max-width: 18px;")
        self.interval_input.setFixedWidth(18)
        self.interval_input.setAlignment(Qt.AlignCenter)
        interval_layout.addWidget(interval_label)
        interval_layout.addWidget(self.interval_input, alignment=Qt.AlignLeft)
        layout.addLayout(interval_layout)

        self.btn_predict = QPushButton("Obtener predicción")
        self.btn_predict.setStyleSheet("background-color: #ff1a1a; color: white; font-size: 18px; padding: 12px 32px; border-radius: 8px;")
        self.btn_predict.setCursor(Qt.PointingHandCursor)
        self.btn_predict.setMinimumWidth(250)
        self.btn_predict.setMaximumWidth(350)
        self.btn_predict.setFont(QFont("Arial", 16, QFont.Bold))
        self.btn_predict.setToolTip("Procesa el video y muestra los resultados")
        self.btn_predict.enterEvent = lambda event: self.btn_predict.setStyleSheet("background-color: #b20000; color: white; font-size: 18px; padding: 12px 32px; border-radius: 8px;")
        self.btn_predict.leaveEvent = lambda event: self.btn_predict.setStyleSheet("background-color: #ff1a1a; color: white; font-size: 18px; padding: 12px 32px; border-radius: 8px;")
        layout.addWidget(self.btn_predict, alignment=Qt.AlignCenter)

        self.setLayout(layout)
        self.btn_select.clicked.connect(lambda: start_callback("select"))
        self.btn_predict.clicked.connect(lambda: start_callback("predict"))

class ResultsWidget(QWidget):
    def __init__(self, frames, predictions, back_callback):
        super().__init__()
        self.frames = frames
        self.predictions = predictions
        self.back_callback = back_callback
        self.idx = 0
        self.setStyleSheet("background-color: #000;")
        pal = self.palette()
        pal.setColor(self.backgroundRole(), Qt.black)
        self.setPalette(pal)
        self.setAutoFillBackground(True)
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignCenter)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("margin-bottom: 20px;")
        self.layout.addWidget(self.img_label)

        self.pred_label = QLabel()
        self.pred_label.setAlignment(Qt.AlignCenter)
        self.pred_label.setStyleSheet("color: white; font-size: 18px; margin-bottom: 30px;")
        self.layout.addWidget(self.pred_label)

        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("←")
        self.btn_prev.setStyleSheet("background-color: #d32f2f; color: white; font-size: 22px; padding: 8px 18px; border-radius: 8px;")
        self.btn_prev.setCursor(Qt.PointingHandCursor)
        self.btn_prev.setMaximumWidth(60)
        self.btn_prev.clicked.connect(self.prev_img)
        nav_layout.addWidget(self.btn_prev)

        self.btn_next = QPushButton("→")
        self.btn_next.setStyleSheet("background-color: #d32f2f; color: white; font-size: 22px; padding: 8px 18px; border-radius: 8px;")
        self.btn_next.setCursor(Qt.PointingHandCursor)
        self.btn_next.setMaximumWidth(60)
        self.btn_next.clicked.connect(self.next_img)
        nav_layout.addWidget(self.btn_next)

        self.btn_back = QPushButton("Volver")
        self.btn_back.setStyleSheet("background-color: #d32f2f; color: white; font-size: 18px; padding: 8px 24px; border-radius: 8px;")
        self.btn_back.setCursor(Qt.PointingHandCursor)
        self.btn_back.setMaximumWidth(120)
        self.btn_back.clicked.connect(self.back_callback)
        nav_layout.addWidget(self.btn_back)

        self.layout.addLayout(nav_layout)
        self.setLayout(self.layout)
        self.update_view()

    def update_view(self):
        pixmap = QPixmap(str(self.frames[self.idx])).scaled(500, 500, Qt.KeepAspectRatio)
        self.img_label.setPixmap(pixmap)
        self.pred_label.setText(self.predictions[self.idx])
        self.btn_prev.setEnabled(self.idx > 0)
        self.btn_next.setEnabled(self.idx < len(self.frames) - 1)

    def next_img(self):
        if self.idx < len(self.frames) - 1:
            self.idx += 1
            self.update_view()

    def prev_img(self):
        if self.idx > 0:
            self.idx -= 1
            self.update_view()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YamaLens")
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet("background-color: #000;")
        self.stacked = QStackedWidget()
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stacked)
        self.setLayout(main_layout)
        self.show()

        self.video_path = None
        self.interval = 1
        self.model_type = 'rfdetr'

        self.welcome = WelcomeWidget(self.handle_welcome_action)
        self.stacked.addWidget(self.welcome)

    def handle_welcome_action(self, action):
        if action == "select":
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(self, "Selecciona un video", "", "Videos (*.mp4 *.avi *.mov)")
            if file_path:
                self.video_path = file_path
                self.welcome.video_label.setText(f"Video seleccionado: {os.path.basename(file_path)}")
        elif action == "predict":
            try:
                self.interval = int(self.welcome.interval_input.text())
            except ValueError:
                self.welcome.interval_input.setText("")
                self.welcome.interval_input.setPlaceholderText("Ingrese un número válido")
                return
            if not self.video_path:
                self.welcome.btn_select.setText("Seleccione un video primero")
                return
            # Leer modelo seleccionado
            selected = self.welcome.model_selector.currentText()
            self.model_type = 'yolo' if selected == 'YOLOv11' else 'rfdetr'
            self.process_video()

    def process_video(self):
        # Eliminar frames previos
        for f in FRAMES_DIR.glob("*.jpg"):
            f.unlink()
        # Extraer frames usando la función centralizada
        extract_frames(self.video_path, FRAMES_DIR, self.interval)
        # Procesar cada frame
        frames = sorted(FRAMES_DIR.glob("*.jpg"))
        if not frames:
            self.welcome.video_label.setText("No se extrajeron frames del video. Verifica el archivo y el intervalo.")
            return
        predictions = [process_frame(f, self.model_type) for f in frames]
        self.results = ResultsWidget(frames, predictions, self.go_back)
        if self.stacked.count() > 1:
            self.stacked.removeWidget(self.stacked.widget(1))
        self.stacked.addWidget(self.results)
        self.stacked.setCurrentWidget(self.results)

    def go_back(self):
        self.stacked.setCurrentWidget(self.welcome)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
