#!/bin/bash

PLATFORM=$1

# Install dependencies
pip install --no-input pyinstaller opencv-python mediapipe pillow pyvirtualcam
pip install --no-input PIL._tkinter_finder || true

if [ "$PLATFORM" = "windows" ]; then
    pyinstaller --noconfirm --onefile --noconsole \
        --name "ASCII-face-cover-windows" \
        --collect-all "mediapipe" \
        --hidden-import="PIL._tkinter_finder" \
        --clean main.py

elif [ "$PLATFORM" = "linux" ]; then
    pip install --no-input pyvirtualcam
    pyinstaller --noconfirm --onefile --noconsole \
        --name "ASCII-face-cover-linux.exe" \
        --collect-all "mediapipe" \
        --hidden-import="PIL._tkinter_finder" \
        --clean main.py

elif [ "$PLATFORM" = "macos" ]; then
    pyinstaller --noconfirm --onefile --noconsole \
        --name "ASCII-face-cover-macos.exe" \
        --collect-all "mediapipe" \
        --hidden-import="PIL._tkinter_finder" \
        --clean main.py

else
    echo "Usage: build.sh [windows|linux|macos]"
    exit 1
fi