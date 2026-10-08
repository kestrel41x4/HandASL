"""
Run inference with a trained YOLO model on images, video, or webcam.

Usage:
    python detect.py                                 # Webcam (default)
    python detect.py --source image.jpg              # Single image
    python detect.py --source path/to/images/        # Directory of images
    python detect.py --source video.mp4              # Video file
    python detect.py --source 0                      # Webcam (device 0)

Options:
    --model PATH     Path to trained weights (default: most recent best.pt)
    --source SOURCE  Image, video, directory, or camera index (default: 0)
    --conf FLOAT     Confidence threshold (default: 0.25)
    --save           Save results to runs/detect/predict/
    --no-show        Do not display results in a window
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent
RUNS_DIR = PROJECT_ROOT / "runs"


def find_best_model():
    """Find the most recent best.pt in runs/detect/."""
    detect_dir = RUNS_DIR / "detect"
    if not detect_dir.exists():
        return None
    candidates = sorted(
        detect_dir.glob("asl_train*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO inference for ASL detection")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model weights (default: most recent best.pt)")
    parser.add_argument("--source", type=str, default="0",
                        help="Image, video, directory, or camera index (default: 0 for webcam)")
    parser.add_argument("--conf", type=float, default=0.85,
                        help="Confidence threshold (default: 0.85)")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated results to runs/detect/predict/")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not display results in a window")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = args.model
    if model_path is None:
        model_path = find_best_model()
        if model_path is None:
            print("ERROR: No trained model found.")
            print("Train a model first: python train.py")
            print("Or specify a model: python detect.py --model path/to/best.pt")
            return
        print(f"Using most recent model: {model_path}")

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return

    model = YOLO(str(model_path))

    source = args.source
    try:
        source = int(source)
        is_webcam = True
    except ValueError:
        is_webcam = False

    show = not args.no_show

    print(f"\nRunning inference:")
    print(f"  Model:      {model_path}")
    print(f"  Source:      {'Webcam ' + str(source) if is_webcam else source}")
    print(f"  Confidence: {args.conf}")
    print(f"  Show:       {show}")
    print(f"  Save:       {args.save}")
    print()

    if is_webcam:
        print("Press 'q' to quit webcam view.")
        cap = cv2.VideoCapture(source)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Center crop to middle 60% to cut out background noise
            h, w = frame.shape[:2]
            crop_ratio = 0.6
            x_off = int(w * (1 - crop_ratio) / 2)
            y_off = int(h * (1 - crop_ratio) / 2)
            cropped = frame[y_off:h - y_off, x_off:w - x_off]

            results = model.predict(
                source=cropped,
                conf=args.conf,
                show=False,
                save=False,
                stream=False,
                verbose=False,
            )
            result = results[0]
            boxes = result.boxes
            if len(boxes) > 0:
                best_idx = int(boxes.conf.argmax())
                best_box = boxes[best_idx]
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                cls_id = int(best_box.cls[0])
                conf = float(best_box.conf[0])
                label = f"{result.names[cls_id]} {conf:.2f}"
                cv2.rectangle(cropped, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cropped, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if show:
                cv2.imshow("ASL Detection", cropped)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cap.release()
        cv2.destroyAllWindows()
    else:
        results = model.predict(
            source=source,
            conf=args.conf,
            show=False,
            save=args.save,
            project=str(RUNS_DIR / "detect") if args.save else None,
            name="predict" if args.save else None,
        )
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                best_idx = int(boxes.conf.argmax())
                best_box = boxes[best_idx]
                cls_id = int(best_box.cls[0])
                conf = float(best_box.conf[0])
                label = result.names[cls_id]
                print(f"\n{result.path}: {label} ({conf:.2f})")
            else:
                print(f"\n{result.path}: No detections")


if __name__ == "__main__":
    main()
