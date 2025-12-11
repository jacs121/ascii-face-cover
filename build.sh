#!/bin/bash

PLATFORM=$1

# Install dependencies
pip install --no-input pyinstaller opencv-python mediapipe pillow pyvirtualcam
pip install --no-input PIL._tkinter_finder || true

if [ "$PLATFORM" = "windows" ]; then
    pyinstaller --noconfirm --onefile --console \
        --name "ASCII-face-cover-windows" \
        --collect-all "mediapipe" \
        --hidden-import="PIL._tkinter_finder" \
        --clean main.py

elif [ "$PLATFORM" = "linux" ]; then
    pyinstaller --noconfirm --onefile --console \
        --name "ASCII-face-cover-linux" \
        --collect-all "mediapipe" \
        --hidden-import="PIL._tkinter_finder" \
        --clean main.py

else
    echo "Usage: build.sh [windows|linux]"
    exit 1
fi