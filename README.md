# ASCII Face Cover

A real-time face anonymization tool that replaces faces with dynamic ASCII expressions. Perfect for streaming, video calls, and privacy protection.

![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-green)

## Features

- Real-time face detection using MediaPipe
- Dynamic ASCII expressions that react to your facial movements
- Virtual camera output for use in any video application
- Custom expression editor with template support
- Background and box texture customization
- Multi-face support (up to 3)
- Head tracking for sharpe angled faces
- Save/Load configuration profiles

## Installation

### From Releases
Download the latest executable for your platform from the [Releases](../../releases) page.

### From Source
```bash
pip install opencv-python mediapipe pillow pyvirtualcam
python main.py
```

## Virtual Camera Setup

### Linux
```bash
sudo apt install v4l2loopback-dkms
sudo modprobe v4l2loopback
```

### macOS / Windows
Install [OBS Studio](https://obsproject.com/) and use OBS Virtual Camera.

## Usage

1. Launch the application
2. Select your camera from the dropdown
3. Adjust detection thresholds as needed
4. Choose an expression mode or create custom ones
5. Enable "Virtual Camera Output" to use in other apps

## Built-in Expressions

| Mode | Example | Description |
|------|---------|-------------|
| AUTO | `'_'` | Dynamic based on your face |
| silly tongue | `:P` `:D` | Tongue out expression |
| wink left | `-_'` | Left eye winking |
| wink right | `'_-` | Right eye winking |
| surprised | `O_O` | Wide eyes |
| dead | `X_X` | X eyes |
| happy | `^v^` | Happy face |
| sad | `v^v` | Sad face |
| tears | `T_T` | Crying |

## Custom Expressions

Click **+ Add** to create custom expressions:

- **Static**: Enter any text (e.g., `UwU`, `OwO`, `:3`)
- **Dynamic**: Use `{left}{mouth}{right}` template with custom character mappings

### Character Mappings
| State | Default | Description |
|-------|---------|-------------|
| left_open | `'` | Left eye open |
| left_closed | `-` | Left eye closed |
| right_open | `'` | Right eye open |
| right_closed | `-` | Right eye closed |
| mouth_open | `o` | Mouth open |
| mouth_closed | `_` | Mouth closed |

## Controls

- **Mirror**: Flip the video horizontally
- **Eye Open Threshold**: Sensitivity for eye open/close detection
- **Surprised Threshold**: Sensitivity for surprised expression
- **Mouth Threshold**: Sensitivity for mouth open detection
- **Head Box Size**: Padding around detected faces

## Configuration Files

Save and load your settings using `.afc` files. These include:
- All threshold settings
- Color choices
- Custom expressions
- Texture paths

## Building from Source

```bash
bash build.sh [windows|linux|macos]
```

## Requirements

- Python 3.8+
- Webcam
- For virtual camera: v4l2loopback (Linux) or OBS (Windows/macOS)

## License

MIT License