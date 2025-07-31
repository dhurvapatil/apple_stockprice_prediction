#!/usr/bin/env python3
"""Setup script for the stock prediction project"""

import subprocess
import sys
from pathlib import Path


def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def create_directories():
    """Create necessary directories"""
    print("Creating project directories...")
    dirs = [
        "data/raw", "data/processed", "data/external",
        "models/trained_models", "models/model_configs",
        "results/plots", "results/reports", "results/predictions",
        "notebooks"
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✓ Directories created")


def main():
    print("Setting up Stock Prediction Project...")

    create_directories()
    install_requirements()

    print("\n✓ Setup completed!")
    print("\nTo run Phase 1:")
    print("python main.py --mode all")


if __name__ == "__main__":
    main()
