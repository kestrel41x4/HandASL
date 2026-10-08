# HandASL - ASL Hand Sign Detection with YOLO

Detect American Sign Language (ASL) hand signs in real-time using YOLO object detection, trained on a dataset from Roboflow.

## Quick Start

### 1. Install uv

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

(Or see the [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/).)

### 2. Install dependencies

uv creates `.venv/` and installs Python 3.12 if needed. Pick a PyTorch build:

```bash
uv sync --extra cpu      # CPU only
uv sync --extra cu128    # NVIDIA GPU (CUDA 12.8)
```

### 3. Set up Roboflow API key

1. Create a free account at [roboflow.com](https://roboflow.com)
2. Get your API key from [Settings > API Keys](https://app.roboflow.com/settings/api-keys)
3. Create a `.env` file in the project root:

```
ROBOFLOW_API_KEY=your_api_key_here
```

### 4. Download the dataset

```bash
uv run download_dataset.py
```

### 5. Train the model

```bash
uv run train.py
```

Custom settings:

```bash
uv run train.py --epochs 100 --batch -1 --model yolov8s.pt
```

### 6. Run inference

```bash
uv run detect.py                          # Webcam (default)
uv run detect.py --source image.jpg       # Single image
uv run detect.py --source video.mp4       # Video file
uv run detect.py --source 0 --save        # Webcam with saved output
```

## Project Structure

```
HandASL/
├── datasets/           # Downloaded dataset (auto-created)
├── runs/               # Training output and predictions (auto-created)
├── train.py            # Model training script
├── detect.py           # Inference/prediction script
├── download_dataset.py # Dataset download from Roboflow
├── pyproject.toml      # Dependencies (managed by uv)
├── uv.lock             # Locked dependency versions
├── .env                # API key (create manually, not in git)
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Using a Different Dataset

Edit the constants in `download_dataset.py`:

```python
WORKSPACE = "your-workspace"
PROJECT = "your-project"
VERSION = 1
```

Browse datasets at [Roboflow Universe](https://universe.roboflow.com).
