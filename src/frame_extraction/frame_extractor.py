"""
frame_extractor.py
extract frames from a video file at specified intervals."""

import cv2
import os
from pathlib import Path

def extract_frames(video_path, output_dir, interval_sec=1):
    """
    Extrae frames de un video cada interval_sec segundos.
    Args:
        video_path (str): Ruta al archivo de video.
        output_dir (str): Carpeta donde se guardarán los frames.
        interval_sec (int): Intervalo en segundos entre cada frame extraído.
    """
    video_path = str(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Verificar permisos de escritura
    test_file = output_dir / "__test_write__.tmp"
    try:
        with open(test_file, "w") as f:
            f.write("test")
        test_file.unlink()
        print(f"writing permises OK: {output_dir}")
    except Exception as e:
        print(f"Don't could write on the folder: {output_dir}\nError: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"don't could open the video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_sec)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS: {fps}, Total frames: {frame_count}, Intervalo: {frame_interval}")

    frame_idx = 0
    saved_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame_file = output_dir / f"frame_{saved_idx:04d}.jpg"
            cv2.imwrite(str(frame_file), frame)
            print(f"Guardado: {frame_file}")
            saved_idx += 1
        frame_idx += 1
    cap.release()
    print(f"Extracción completada. Total de frames guardados: {saved_idx}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extrae frames de un video.")
    parser.add_argument("video_path", type=str, help="Ruta al archivo de video")
    parser.add_argument(
        "output_dir",
        type=str,
        nargs='?',
        default=str(Path(__file__).parent.parent / "frames-output"),
        help="Carpeta de salida para los frames (por defecto: src/frames-output)"
    )
    parser.add_argument("--interval", type=int, default=1, help="Intervalo en segundos entre frames")
    args = parser.parse_args()
    extract_frames(args.video_path, args.output_dir, args.interval)
