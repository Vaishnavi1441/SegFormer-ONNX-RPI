"""
Model Training Script for Custom Multi-Modal UAV Dataset (RTX 4050 GPU)
Trains YOLO11n / YOLOv8n on RGB or Thermal datasets, then exports to ONNX & NCNN for Pi 5.
"""

import os
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Trainer")

def train(data_yaml: str, epochs: int = 50, batch: int = 16, imgsz: int = 640, model_type: str = "yolo11n.pt"):
    try:
        from ultralytics import YOLO
        logger.info(f"Initializing YOLO model: {model_type} on CUDA (RTX 4050)...")
        model = YOLO(model_type)

        logger.info(f"Starting training with dataset: {data_yaml}")
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=0, # GPU index 0 (RTX 4050)
            plots=True
        )

        logger.info("Training complete! Exporting to ONNX for Raspberry Pi 5 CPU / ARM NEON...")
        onnx_path = model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True)
        logger.info(f"ONNX Model saved to: {onnx_path}")

        # Optional NCNN export
        try:
            ncnn_path = model.export(format="ncnn")
            logger.info(f"NCNN Model saved to: {ncnn_path}")
        except Exception as e:
            logger.warning(f"NCNN export skipped: {e}")

    except ImportError:
        logger.error("Ultralytics not installed. Install with: pip install ultralytics")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model on RTX 4050")
    parser.add_argument("--data", type=str, default="dataset.yaml", help="Path to dataset yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model weight")
    args = parser.parse_args()

    train(args.data, epochs=args.epochs, model_type=args.model)
