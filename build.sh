#!/usr/bin/env bash
# Build script for Render

set -o errexit

# Install system dependencies (Tesseract OCR)
apt-get update && apt-get install -y tesseract-ocr

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
