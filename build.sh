#!/bin/bash

PLATFORM=$1

# Install dependencies
pip install pyinstaller opencv-python mediapipe pillow pyvirtualcam
pip install PIL._tkinter_finder || true

if [ "$PLATFORM" = "windows" ]; then
    pyinstaller --noconfirm --onefile --console --name "ASCII face cover - windows" --collect-all "mediapipe" --hidden-import="PIL._tkinter_finder" --clean  "/home/nitzan/Documents/my shish/programming/python/ascii emoji face cover/main.py"

elif [ "$PLATFORM" = "linux" ]; then
    pyinstaller --noconfirm --onefile --console --name "ASCII face cover - linux" --collect-all "mediapipe" --hidden-import="PIL._tkinter_finder" --clean  "/home/nitzan/Documents/my shish/programming/python/ascii emoji face cover/main.py"

else
    echo "Usage: build.sh [windows|linux]"
    exit 1
fi
