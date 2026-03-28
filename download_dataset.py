"""
Download ASL dataset from Roboflow.

Usage:
    python download_dataset.py

Requires ROBOFLOW_API_KEY in .env file or as environment variable.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow


def main():
    load_dotenv()

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("ERROR: ROBOFLOW_API_KEY not found.")
        print("Create a .env file with: ROBOFLOW_API_KEY=your_key_here")
        print("Get your key at: https://app.roboflow.com/settings/api-keys")
        return

    # --- Dataset Configuration ---
    # American Sign Language Letters dataset by David Lee
    # https://universe.roboflow.com/david-lee-d0rhs/american-sign-language-letters
    # 26 classes (A-Z), ~1700+ images with bounding box annotations
    WORKSPACE = "david-lee-d0rhs"
    PROJECT = "american-sign-language-letters"
    VERSION = 1
    FORMAT = "yolov8"

    dataset_dir = Path(__file__).parent / "datasets"
    dataset_dir.mkdir(exist_ok=True)

    print("Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)

    print(f"Downloading {PROJECT} v{VERSION} in {FORMAT} format...")
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download(
        model_format=FORMAT,
        location=str(dataset_dir / PROJECT),
    )

    print(f"\nDataset downloaded to: {dataset.location}")
    print(f"data.yaml location:   {dataset.location}/data.yaml")
    print("\nYou can now run: python train.py")


if __name__ == "__main__":
    main()
