"""
Train a YOLO model on the ASL hand sign detection dataset.

Usage:
    python train.py                           # Train with defaults
    python train.py --epochs 50               # Custom epochs
    python train.py --model yolov8s.pt        # Use a larger model
    python train.py --resume                  # Resume interrupted training

Requires the dataset to be downloaded first:
    python download_dataset.py
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "american-sign-language-letters"
DATA_YAML = DATASET_DIR / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "runs"

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_EPOCHS = 50
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_PATIENCE = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO on ASL dataset")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Pretrained model to start from (default: {DEFAULT_MODEL})")
    parser.add_argument("--data", type=str, default=str(DATA_YAML),
                        help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help=f"Input image size (default: {DEFAULT_IMGSZ})")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help=f"Batch size, use -1 for auto (default: {DEFAULT_BATCH})")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                        help=f"Early stopping patience (default: {DEFAULT_PATIENCE})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        print("Run 'python download_dataset.py' first to download the dataset.")
        return

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"\nStarting training:")
    print(f"  Model:      {args.model}")
    print(f"  Dataset:    {args.data}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {args.batch}")
    print(f"  Patience:   {args.patience}")
    print()

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=str(RUNS_DIR / "detect"),
        name="asl_train",
        exist_ok=False,
        save=True,
        save_period=-1,
        plots=True,
        verbose=True,
    )

    best_weight = RUNS_DIR / "detect" / "asl_train" / "weights" / "best.pt"
    print(f"\nTraining complete!")
    print(f"Best weights saved to: {best_weight}")
    print(f"Run 'python detect.py --model {best_weight}' to test inference.")


if __name__ == "__main__":
    main()
