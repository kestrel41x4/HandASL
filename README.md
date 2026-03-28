# HandASL - ASL Hand Sign Detection with YOLO

Detect American Sign Language (ASL) hand signs in real-time using YOLO object detection, trained on a dataset from Roboflow.

## Quick Start

### 1. Create virtual environment

```bash
py -3.12 -m venv venv
venv\Scripts\activate
pip install --upgrade pip
```

### 2. (Optional) Install PyTorch with GPU support

If you have an NVIDIA GPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Skip this step for CPU-only training.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up Roboflow API key

1. Create a free account at [roboflow.com](https://roboflow.com)
2. Get your API key from [Settings > API Keys](https://app.roboflow.com/settings/api-keys)
3. Create a `.env` file in the project root:

```
ROBOFLOW_API_KEY=your_api_key_here
```

### 5. Download the dataset

```bash
python download_dataset.py
```

### 6. Train the model

```bash
python train.py
```

Custom settings:

```bash
python train.py --epochs 100 --batch -1 --model yolov8s.pt
```

### 7. Run inference

```bash
python detect.py                          # Webcam (default)
python detect.py --source image.jpg       # Single image
python detect.py --source video.mp4       # Video file
python detect.py --source 0 --save        # Webcam with saved output
```

## Project Structure

```
HandASL/
├── datasets/           # Downloaded dataset (auto-created)
├── runs/               # Training output and predictions (auto-created)
├── train.py            # Model training script
├── detect.py           # Inference/prediction script
├── download_dataset.py # Dataset download from Roboflow
├── requirements.txt    # Python dependencies
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
