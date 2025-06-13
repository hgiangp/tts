#!/bin/bash

set -e

# Check for pyenv
if ! command -v pyenv &> /dev/null; then
  echo "[INFO] Installing pyenv..."
  brew install pyenv
fi

echo "[INFO] Installing Python 3.10.13 using pyenv..."
pyenv install -s 3.10.13
pyenv local 3.10.13

echo "[INFO] Creating virtual environment..."
python3 -m venv .venv

echo "[INFO] Activating virtual environment..."
source .venv/bin/activate

echo "[INFO] Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "[INFO] Installing Python packages..."
pip install torch==2.5.1 numpy soundfile mecab-python3 unidic-lite
pip install "TTS[ja]" transformers huggingface_hub

echo "[INFO] Downloading MMS-TTS Vietnamese model..."
python3 download_tts_models.py


echo "[DONE] Setup complete!"
echo
echo "[USAGE] To activate your environment next time:"
echo "  source .venv/bin/activate"
