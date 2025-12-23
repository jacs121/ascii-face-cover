# main.py - improved stable detection version
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, messagebox
import subprocess
import platform
from PIL import Image, ImageTk
import threading
import time
from collections import deque
import zlib
import pickle

# Check v4l2loopback on Linux
v4l2_message = None
if platform.system() == "Linux":
    result = subprocess.run(["lsmod"], capture_output=True, text=True)
    if "v4l2loopback" not in result.stdout:
        try:
            subprocess.run(["pkexec", "modprobe", "v4l2loopback"], check=True)
        except subprocess.CalledProcessError:
            v4l2_message = "Failed to load v4l2loopback.\n\nRun in terminal:\nsudo apt install v4l2loopback-dkms"
        except FileNotFoundError:
            v4l2_message = "pkexec not found. Run in terminal:\nsudo modprobe v4l2loopback"
else:
    v4l2_message = "Failed to load v4l2loopback.\n\nyou need to use OBS Virtual Camera"

# Try to import pyvirtualcam for virtual camera output
try:
    import pyvirtualcam
    VCAM_AVAILABLE = True
except Exception:
    VCAM_AVAILABLE = False

# ==================== CONFIG ====================
class Config:
    def __init__(self):
        self.mirror = True
        self.ear_threshold = 0.175
        self.mouth_threshold = 6
        self.surprised_threshold = 0.275
        self.smile_threshold = 2
        self.head_padding = 0.3
        self.bg_texture_path = None
        self.box_texture_path = None
        self.bg_color = (50, 50, 50)
        self.box_color = (0, 0, 0)
        self.text_color = (255, 255, 255)
        self.special_mode = "AUTO"
        self.virtual_cam_enabled = False
        self.custom_expressions = {}
    
    def save(self, filepath: str):
        data = self.__dict__
        pickled_data = pickle.dumps(data)

        compressed_data = zlib.compress(pickled_data)
        open(filepath, "wb").write(compressed_data)

    def load(self, filepath, app: 'AsciiFaceCoverApp'):
        newConfig = defaultConfig.__dict__
        try:
            decompressed_data = zlib.decompress(open(filepath, "rb").read())
            loaded_data = pickle.loads(decompressed_data)
            
            if not isinstance(loaded_data, dict):
                print("ERROR: invalid data structure")
                return False, "ERROR: invalid data structure"
            
            # set config values
            for key, value in loaded_data.items():
                if key not in newConfig.keys():
                    print(f"ERROR: cannot set fake data: {key}")
                    return False, f"ERROR: cannot set fake data: {key}"
                if not isinstance(value, type(newConfig[key])):
                    print(f"ERROR: invalid data type for {key}: {type(value)}")
                    return False, f"ERROR: invalid data type for {key}: {type(value)}"
                newConfig[key] = value
            app.reset_config(newConfig, forced={}, ignore=[])
            return True, "loaded new settings"
        except Exception as e:
            print(str(e))
            return False, str(e)


config = Config()
defaultConfig = Config()

# ==================== MEDIAPIPE SETUP ====================
# FaceMesh kept single instance inside worker thread (created in worker)
mp_face_mesh = None
FACE_MESH_MAX_FACES = 3

# MediaPipe Pose for head detection when face isn't visible
mp_pose = None

# Landmark groups used:
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
UPPER_LIP_IDX = [13, 312]   # use a couple points for averaging
LOWER_LIP_IDX = [14, 17]    # use a couple points for averaging
CHEEK_LEFT = 234
CHEEK_RIGHT = 454
NOSE_TIP = 1
CHIN = 152

# ==================== UTILS ====================
def safe_read_image(path):
    if not path:
        return None
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        return img
    except Exception:
        return None

def clamp_point(x, minv, maxv):
    return max(minv, min(maxv, x))

# EAR using numpy array of shape (N,2) where coords in *pixel* space
def calc_ear_pts(pts):
    # pts: array of 6 points (x,y) in pixels
    if pts.shape[0] < 6:
        return 0.0
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h) if h > 1e-6 else 0.0

# Cached rotated text images to avoid regen each frame
_text_cache = {}

def get_cached_text_image(text, font_scale, thickness, angle, color_bgr):
    key = (text, round(font_scale,2), thickness, round(angle,1), color_bgr)
    if key in _text_cache:
        return _text_cache[key]
    font = cv2.FONT_HERSHEY_SIMPLEX
    ts = cv2.getTextSize(text, font, font_scale, thickness)[0]
    w, h = ts[0] + 12, ts[1] + 12
    txt_img = np.zeros((h, w, 4), dtype=np.uint8)
    cv2.putText(txt_img, text, (6, h - 6), font, font_scale, (*color_bgr, 255), thickness, cv2.LINE_AA)
    # rotate
    M = cv2.getRotationMatrix2D((w//2, h//2), -angle, 1.0)
    cos = abs(M[0,0]); sin = abs(M[0,1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0,2] += (new_w - w)/2
    M[1,2] += (new_h - h)/2
    rotated = cv2.warpAffine(txt_img, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    _text_cache[key] = rotated
    return rotated

def alpha_blend(dst, overlay, x, y):
    # overlay is RGBA numpy array
    h, w = overlay.shape[:2]
    if x < 0:
        overlay = overlay[:, -x:]
        w = overlay.shape[1]
        x = 0
    if y < 0:
        overlay = overlay[-y:, :]
        h = overlay.shape[0]
        y = 0
    if x + w > dst.shape[1]:
        overlay = overlay[:, :dst.shape[1]-x]
        w = overlay.shape[1]
    if y + h > dst.shape[0]:
        overlay = overlay[:dst.shape[0]-y, :]
        h = overlay.shape[0]
    if h <= 0 or w <= 0:
        return
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    dst_slice = dst[y:y+h, x:x+w].astype(np.float32)
    overlay_rgb = overlay[:, :, :3].astype(np.float32)
    blended = overlay_rgb * alpha + dst_slice * (1 - alpha)
    dst[y:y+h, x:x+w] = blended.astype(np.uint8)

# ==================== STABLE DETECTION WORKER ====================
class DetectionWorker(threading.Thread):
    def __init__(self, cap_idx=0):
        super().__init__(daemon=True)
        global mp_face_mesh, mp_pose
        self.cap_idx = cap_idx
        self.cap = None
        self.running = False
        self.output_frame = None
        self.frame_lock = threading.Lock()
        self.last_process_time = 0.0
        
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        mp_pose = mp.solutions.pose
        

        # Per-face tracking data (keyed by face index)
        self.face_data = {}
        self.alpha_landmark = 0.25

        # mediapipe face mesh instance
        self.face_mesh = None

    def open_camera(self, idx):
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
        self.cap = cv2.VideoCapture(idx)
        # try to set a modest resolution for stability
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(0.1)

    def run(self):
        # create MediaPipe FaceMesh here (keeps it off the UI thread)
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                               max_num_faces=FACE_MESH_MAX_FACES,
                                               refine_landmarks=True,
                                               min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)
        self.pose = mp_pose.Pose(static_image_mode=False,
                                 model_complexity=0,
                                 min_detection_confidence=0.3,
                                 min_tracking_confidence=0.3)
        self.open_camera(self.cap_idx)
        self.running = True
        while self.running:
            start = time.time()
            ret, frame = self.cap.read() if self.cap else (False, None)
            if not ret or frame is None:
                # small sleep to avoid busy-looping on capture errors
                time.sleep(0.02)
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # process with mediapipe (this can take variable time)
            try:
                results = self.face_mesh.process(rgb)
            except Exception:
                results = None

            # prepare output base image (use background texture if available)
            if config.bg_texture_path:
                bg = safe_read_image(config.bg_texture_path)
                if bg is not None:
                    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
                    output = bg.copy()
                else:
                    output = np.full((h, w, 3), config.bg_color, dtype=np.uint8)
            else:
                output = frame.copy()

            mediapipe_boxes = []  # Track mediapipe detections for Haar fallback
            if results and results.multi_face_landmarks:
                for face_idx, result in enumerate(results.multi_face_landmarks):
                    landmarks = result.landmark

                    # convert landmarks to numpy array (x,y,z) in normalized coords
                    pts = np.array([[l.x, l.y, l.z] for l in landmarks], dtype=np.float32)

                    # convert to pixel coords for numerical stability
                    pts_px = pts.copy()
                    pts_px[:, 0] *= w
                    pts_px[:, 1] *= h

                    # sanity checks
                    xs = pts_px[:, 0]
                    ys = pts_px[:, 1]
                    if np.isnan(xs).any() or np.isnan(ys).any():
                        continue
                    
                    spread_x = xs.max() - xs.min()
                    spread_y = ys.max() - ys.min()
                    if spread_x < 10 or spread_y < 10:
                        continue
                    
                    # Get or create per-face tracking data
                    if face_idx not in self.face_data:
                        self.face_data[face_idx] = {
                            'landmark_smooth': None,
                            'left_ear_hist': deque(maxlen=4),
                            'right_ear_hist': deque(maxlen=4),
                            'mouth_hist': deque(maxlen=9),
                            'yaw_hist': deque(maxlen=7),
                            'pitch_hist': deque(maxlen=7),
                            'roll_hist': deque(maxlen=7),
                            'cal_frames': 0,
                            'cal_left_max': 0.0,
                            'cal_right_max': 0.0,
                            'prev_left_state': True,
                            'prev_right_state': True,
                        }
                    fd = self.face_data[face_idx]
                    
                    # Exponential smoothing on landmarks
                    if fd['landmark_smooth'] is None:
                        fd['landmark_smooth'] = pts_px.copy()
                    else:
                        a = self.alpha_landmark
                        fd['landmark_smooth'] = a * pts_px + (1 - a) * fd['landmark_smooth']

                    lm = fd['landmark_smooth']

                    # EAR calculation
                    left_pts = lm[LEFT_EYE_IDX][:, :2]
                    right_pts = lm[RIGHT_EYE_IDX][:, :2]
                    left_ear = calc_ear_pts(left_pts)
                    right_ear = calc_ear_pts(right_pts)

                    fd['left_ear_hist'].append(left_ear)
                    fd['right_ear_hist'].append(right_ear)

                    def weighted_mean(dq):
                        if len(dq) == 0:
                            return 0.0
                        arr = np.array(dq)
                        weights = np.linspace(1.0, 2.0, len(arr))
                        return float(np.sum(arr * weights) / np.sum(weights))
                    
                    left_smooth = weighted_mean(fd['left_ear_hist'])
                    right_smooth = weighted_mean(fd['right_ear_hist'])

                    # Calibration
                    if fd['cal_frames'] < 40:
                        fd['cal_left_max'] = max(fd['cal_left_max'], left_smooth)
                        fd['cal_right_max'] = max(fd['cal_right_max'], right_smooth)
                        fd['cal_frames'] += 1
                    
                    # Calculate pitch for threshold adjustments
                    forehead_y = lm[10, 1]
                    chin_y = lm[CHIN, 1]
                    nose_y_early = lm[NOSE_TIP, 1]
                    upper_face = nose_y_early - forehead_y
                    lower_face = chin_y - nose_y_early
                    face_ratio = upper_face / lower_face if lower_face > 1 else 0.8
                    
                    # Proportional pitch adjustment with dead zone (0.7-0.9 = neutral)
                    if face_ratio < 0.7:
                        pitch_deviation = face_ratio - 0.7  # negative when looking up
                    elif face_ratio > 0.9:
                        pitch_deviation = face_ratio - 0.9  # positive when looking down
                    else:
                        pitch_deviation = 0  # dead zone - no adjustment
                    
                    # Eye threshold: raise when down, lower when up
                    base_thresh = 0.70 + pitch_deviation * 1.0
                    base_thresh = max(0.45, min(0.85, base_thresh))
                    
                    # Per-eye yaw adjustment (nose offset from center)
                    nose_x = lm[NOSE_TIP, 0]
                    face_center_x = (lm[CHEEK_LEFT, 0] + lm[CHEEK_RIGHT, 0]) / 2
                    yaw_offset = (nose_x - face_center_x) / (spread_x * 0.5) if spread_x > 1 else 0
                    
                    # When looking right (positive yaw): left eye farther, needs lower threshold
                    # When looking left (negative yaw): right eye farther, needs lower threshold
                    left_thresh = base_thresh - max(0, yaw_offset) * 0.15
                    right_thresh = base_thresh + min(0, yaw_offset) * 0.15
                    left_thresh = max(0.45, min(0.85, left_thresh))
                    right_thresh = max(0.45, min(0.85, right_thresh))
                    
                    def is_open_left(smooth_val, cal_max):
                        if cal_max > 1e-6:
                            return smooth_val > (cal_max * left_thresh)
                        return smooth_val > config.ear_threshold
                    
                    def is_open_right(smooth_val, cal_max):
                        if cal_max > 1e-6:
                            return smooth_val > (cal_max * right_thresh)
                        return smooth_val > config.ear_threshold

                    left_open = is_open_left(left_smooth, fd['cal_left_max'])
                    right_open = is_open_right(right_smooth, fd['cal_right_max'])

                    # mouth detections
                    top_lip_y = np.mean(lm[UPPER_LIP_IDX][:, 1])
                    bottom_lip_y = np.mean(lm[LOWER_LIP_IDX][:, 1])
                    mouth_dist = (bottom_lip_y - top_lip_y)
                    fd['mouth_hist'].append(mouth_dist)
                    mouth_s = float(np.mean(fd['mouth_hist']))
                    
                    # Mouth pitch: looking down compresses appearance, boost detection value
                    if pitch_deviation > 0:  # looking down
                        mouth_pitch_factor = 1.0 + pitch_deviation * 1.5  # Boost to compensate
                    else:
                        mouth_pitch_factor = 1.0
                    
                    # Detect smile using mouth corners (landmarks 61, 291)
                    left_corner_y = lm[61, 1]
                    right_corner_y = lm[291, 1]
                    mouth_center_y = (top_lip_y + bottom_lip_y) / 2
                    smile_detected = (left_corner_y < mouth_center_y - config.smile_threshold) and (right_corner_y < mouth_center_y - config.smile_threshold)
                    
                    face_height = spread_y if spread_y > 1 else 1.0
                    mouth_open = (mouth_s * mouth_pitch_factor) > (config.mouth_threshold * 0.01 * face_height)
                    
                    if smile_detected and not mouth_open:
                        mouth_char = "v"
                    elif mouth_open:
                        mouth_char = "o"
                    else:
                        mouth_char = "_"

                    # Head pose - yaw calculation
                    nose_x = lm[NOSE_TIP, 0]
                    face_center_x = (lm[CHEEK_LEFT, 0] + lm[CHEEK_RIGHT, 0]) / 2
                    yaw_offset = (nose_x - face_center_x) * 0.6
                    
                    # Pitch offset for text positioning (face_ratio already calculated above)
                    eye_level = (lm[33, 1] + lm[263, 1]) / 2
                    pitch_offset = (nose_y_early - eye_level - spread_y * 0.15) * 0.6
                    
                    left_smooth_adj = left_smooth
                    right_smooth_adj = right_smooth

                    eye_left = lm[33][:2]
                    eye_right = lm[263][:2]
                    dy = (eye_right[1] - eye_left[1])
                    dx = (eye_right[0] - eye_left[0])
                    roll_deg = float(np.degrees(np.arctan2(dy, dx)))
                    
                    fd['roll_hist'].append(roll_deg)
                    fd['yaw_hist'].append(yaw_offset)
                    fd['pitch_hist'].append(pitch_offset)
                    
                    roll_s = float(np.mean(fd['roll_hist']))
                    yaw_s = float(np.mean(fd['yaw_hist']))
                    pitch_s = float(np.mean(fd['pitch_hist']))

                    # Hysteresis
                    if left_open and not fd['prev_left_state']:
                        left_open = left_smooth_adj > (fd['cal_left_max'] * 0.75)
                    elif not left_open and fd['prev_left_state']:
                        left_open = left_smooth_adj < (fd['cal_left_max'] * 0.60)
                    fd['prev_left_state'] = left_open

                    if right_open and not fd['prev_right_state']:
                        right_open = right_smooth_adj > (fd['cal_right_max'] * 0.75)
                    elif not right_open and fd['prev_right_state']:
                        right_open = right_smooth_adj < (fd['cal_right_max'] * 0.60)
                    fd['prev_right_state'] = right_open

                    # Expression
                    left_sym = "'" if left_open else "-"
                    right_sym = "'" if right_open else "-"
                    # mouth_char already set above with smile detection
                    # Adjust surprised threshold only outside dead zone
                    surprised_thresh = config.surprised_threshold * (1.0 + pitch_deviation * 2.0)
                    surprised_thresh = max(config.surprised_threshold * 0.8, surprised_thresh)
                    if left_smooth_adj > surprised_thresh and right_smooth_adj > surprised_thresh:
                        left_sym = right_sym = "O"

                    emoji = None
                    if config.special_mode == "AUTO":
                        emoji = f"{left_sym}{mouth_char}{right_sym}"
                    elif config.special_mode.startswith("custom:"):
                        name = config.special_mode.replace("custom:", "")
                        if name in config.custom_expressions:
                            expr = config.custom_expressions[name]
                            template = expr.get("template", "{left}{mouth}{right}")
                            if "{left}" in template or "{mouth}" in template or "{right}" in template:
                                left_c = expr.get("left_open", "'") if left_open else expr.get("left_closed", "-")
                                right_c = expr.get("right_open", "'") if right_open else expr.get("right_closed", "-")
                                if mouth_open:
                                    mouth_c = expr.get("mouth_open", "o")
                                elif smile_detected:
                                    mouth_c = expr.get("mouth_smile", "v")
                                else:
                                    mouth_c = expr.get("mouth_closed", "_")
                                emoji = template.replace("{left}", left_c).replace("{mouth}", mouth_c).replace("{right}", right_c)
                            else:
                                emoji = template  # Static text
                        else:
                            emoji = f"{left_sym}{mouth_char}{right_sym}"
                    else:
                        mode = config.special_mode
                        if mode == "silly_tongue": emoji = (":" if left_sym == right_sym == "'" else "X")+("D" if mouth_open else "P")
                        elif mode == "wink_left": emoji = f"-{mouth_char}{right_sym}"
                        elif mode == "wink_right": emoji = f"{left_sym}{mouth_char}-"
                        elif mode == "surprised": emoji = "O" + mouth_char + "O"
                        elif mode == "dead": emoji = "X" + mouth_char + "X"
                        elif mode == "happy": emoji = (left_sym.replace("'", "^") + mouth_char.replace("_","v") + right_sym.replace("'", "^"))
                        elif mode == "sad": emoji = (left_sym.replace("'", "v") + mouth_char.replace("_","^") + right_sym.replace("'", "v"))
                        elif mode == "tears": emoji = "T" + mouth_char + "T"
                        else: emoji = f"{left_sym}{mouth_char}{right_sym}"

                    # Draw box
                    fw = spread_x; fh = spread_y
                    x1 = int(max(0, xs.min()))
                    x2 = int(min(w, xs.max()))
                    y1 = int(max(0, ys.min()))
                    y2 = int(min(h, ys.max()))
                    
                    # Calculate text size first to ensure box fits it
                    # Adjust font scale based on emoji character count
                    char_count = len(emoji)
                    font_scale = max(0.9, fw / (45 + (char_count - 3) * 8)) if char_count > 3 else max(0.9, fw / 45)
                    thickness = max(2, int(fw / 35))
                    # Get actual rotated text image size
                    text_img = get_cached_text_image(emoji, font_scale, thickness, angle=roll_s, color_bgr=config.text_color)
                    # Calculate extra padding based on actual character widths
                    extra_char_pad = 0
                    for c in emoji:
                        char_w = cv2.getTextSize(c, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
                        extra_char_pad += int(char_w * 0.3)
                    text_w = text_img.shape[1] + 20 + extra_char_pad
                    text_h = text_img.shape[0] + 20
                    
                    pad_x = int(fw * config.head_padding)
                    pad_top = int(fh * config.head_padding * 1.5)
                    pad_bottom = int(fh * config.head_padding * 0.5)
                    
                    # Expand if text is larger than face box
                    face_w = x2 - x1
                    face_h = y2 - y1
                    if text_w > face_w + 2 * pad_x:
                        pad_x = (text_w - face_w) // 2 + 10
                    if text_h > face_h + pad_top + pad_bottom:
                        extra = (text_h - face_h - pad_top - pad_bottom) // 2 + 10
                        pad_top += extra
                        pad_bottom += extra
                    bx1 = max(0, x1 - pad_x)
                    by1 = max(0, y1 - pad_top)
                    bx2 = min(w, x2 + pad_x)
                    by2 = min(h, y2 + pad_bottom)
                    mediapipe_boxes.append((bx1, by1, bx2, by2))
                    if config.box_texture_path:
                        tex = safe_read_image(config.box_texture_path)
                        if tex is not None:
                            try:
                                tex = cv2.resize(tex, (bx2-bx1, by2-by1))
                                output[by1:by2, bx1:bx2] = tex
                            except Exception:
                                cv2.rectangle(output, (bx1, by1), (bx2, by2), config.box_color, -1)
                        else:
                            cv2.rectangle(output, (bx1, by1), (bx2, by2), config.box_color, -1)
                    else:
                        cv2.rectangle(output, (bx1, by1), (bx2, by2), config.box_color, -1)

                    # Draw emoji with head position offset
                    center_x = int((x1 + x2)/2 + yaw_s)
                    center_y = int((y1 + y2)/2 + pitch_s)
                    alpha_blend(output, text_img, center_x - text_img.shape[1]//2, center_y - text_img.shape[0]//2)
            # Fallback: Use MediaPipe Pose for heads not detected by face mesh
            try:
                pose_results = self.pose.process(rgb)
                if pose_results.pose_landmarks:
                    lm = pose_results.pose_landmarks.landmark
                    # Head landmarks: nose(0), left_eye(2), right_eye(5), left_ear(7), right_ear(8)
                    head_points = []
                    for idx in [0, 2, 5, 7, 8]:
                        if lm[idx].visibility > 0.3:
                            head_points.append((int(lm[idx].x * w), int(lm[idx].y * h)))
                    
                    if len(head_points) >= 2:
                        xs = [p[0] for p in head_points]
                        ys = [p[1] for p in head_points]
                        cx, cy = sum(xs) // len(xs), sum(ys) // len(ys)
                        
                        # Check if this head overlaps with any mediapipe face box
                        covered = False
                        for (mx1, my1, mx2, my2) in mediapipe_boxes:
                            if mx1 <= cx <= mx2 and my1 <= cy <= my2:
                                covered = True
                                break
                        
                        if not covered:
                            # Estimate head size from shoulder width or use default
                            left_shoulder = lm[11]
                            right_shoulder = lm[12]
                            if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                                shoulder_w = abs(left_shoulder.x - right_shoulder.x) * w
                                head_size = int(shoulder_w * 0.6)
                            else:
                                head_size = int(min(w, h) * 0.15)
                            
                            pad = int(head_size * config.head_padding)
                            bx1 = max(0, cx - head_size//2 - pad)
                            by1 = max(0, cy - head_size//2 - pad)
                            bx2 = min(w, cx + head_size//2 + pad)
                            by2 = min(h, cy + head_size//2 + pad)
                            cv2.rectangle(output, (bx1, by1), (bx2, by2), config.box_color, -1)
            except Exception:
                pass

            # mirror for display if required
            if config.mirror:
                output = cv2.flip(output, 1)

            # store output frame (thread-safe)
            with self.frame_lock:
                self.output_frame = output

            # small throttle depending on processing time to avoid camera backlog
            elapsed = time.time() - start
            self.last_process_time = elapsed
            if elapsed < 1/25:
                time.sleep(max(0, 1/25 - elapsed))

        # cleanup
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
        if self.face_mesh:
            try:
                self.face_mesh.close()
            except:
                pass

    def get_frame(self):
        with self.frame_lock:
            if self.output_frame is not None:
                return self.output_frame.copy()
            else:
                return None

    def stop(self):
        self.running = False

# ==================== UI APP ====================
def get_available_cameras():
    cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to get camera name on Linux
            name = f"Camera {i}"
            if platform.system() == "Linux":
                try:
                    import subprocess
                    result = subprocess.run(["v4l2-ctl", "-d", f"/dev/video{i}", "--info"], 
                                          capture_output=True, text=True)
                    for line in result.stdout.split('\n'):
                        if 'Card type' in line:
                            card_name = line.split(':')[1].strip()
                            # Skip v4l2loopback virtual cameras
                            if 'v4l2loopback' in card_name.lower() or 'dummy' in card_name.lower():
                                cap.release()
                                break
                            name = f"{i}: {card_name}"
                            break
                    else:
                        cameras.append((i, name))
                        cap.release()
                        continue
                except:
                    cameras.append((i, name))
                cap.release()
            else:
                cameras.append((i, name))
                cap.release()
    return cameras if cameras else [(0, "Camera 0")]

class CustomExpressionDialog:
    def __init__(self, parent, name="", data=None):
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Custom Expression")
        # self.dialog.geometry("350x280")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        data = data or {}
        
        ttk.Label(self.dialog, text="Name:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.name_var = tk.StringVar(value=name)
        ttk.Entry(self.dialog, textvariable=self.name_var, width=25).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(self.dialog, text="Template ({left}{mouth}{right} or static characters):").grid(row=1, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        self.template_var = tk.StringVar(value=data.get("template", "{left}{mouth}{right}"))
        ttk.Entry(self.dialog, textvariable=self.template_var, width=30).grid(row=2, column=0, columnspan=2, padx=5, pady=2)
        
        ttk.Separator(self.dialog, orient='horizontal').grid(row=3, column=0, columnspan=2, sticky='ew', pady=10)
        ttk.Label(self.dialog, text="Character mappings:").grid(row=4, column=0, columnspan=2, sticky='w', padx=5)
        
        fields = [("left_open", "'"), ("left_closed", "-"), ("right_open", "'"), 
                  ("right_closed", "-"), ("mouth_open", "o"), ("mouth_closed", "_"), ("mouth_smile", "v")]
        self.char_vars = {}
        for i, (field, default) in enumerate(fields):
            row = 5 + i // 2
            col = i % 2
            frame = ttk.Frame(self.dialog)
            frame.grid(row=row, column=col, sticky='w', padx=5, pady=1)
            ttk.Label(frame, text=f"{field.replace('_', ' ')}:").pack(side='left')
            self.char_vars[field] = tk.StringVar(value=data.get(field, default))
            ttk.Entry(frame, textvariable=self.char_vars[field], width=5).pack(side='left', padx=2)
        
        btn_frame = ttk.Frame(self.dialog)
        btn_frame.grid(row=9, column=0, columnspan=2, pady=15)
        ttk.Button(btn_frame, text="Save", command=self.save).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.dialog.destroy).pack(side='left', padx=5)
        
        self.dialog.wait_window()
    
    def save(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showerror("Error", "Name is required")
            return
        self.result = {
            "name": name,
            "data": {
                "template": self.template_var.get(),
                **{k: v.get() for k, v in self.char_vars.items()}
            }
        }
        self.dialog.destroy()

class AsciiFaceCoverApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ASCII Face Cover")
        self.root.geometry("900x600")
        self.root.minsize(700, 500)
        # Show loading popup
        self.loading_popup = tk.Toplevel(self.root)
        self.loading_popup.title("Loading...")
        self.loading_popup.geometry("250x80")
        self.loading_popup.resizable(False, False)
        self.loading_popup.transient(self.root)
        self.loading_popup.grab_set()
        ttk.Label(self.loading_popup, text="Initializing camera and models...", font=("", 11)).pack(expand=True)
        self.loading_popup.update()

        self.available_cameras = get_available_cameras()
        self.current_cam_idx = self.available_cameras[0][0]
        self.fullscreen_cam = None

        # Detection worker thread
        self.worker = DetectionWorker(cap_idx=self.current_cam_idx)
        self.worker.start()

        self.vcam = None
        self.cam_aspect = 4/3
        self.loading_done = False

        self.setup_ui()
        self.update_video()

    def setup_ui(self):
        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill='both', expand=True)

        # Controls
        ctrl_frame = ttk.Frame(self.main_frame)
        ctrl_frame.pack(side='right', fill='y', padx=5, pady=5)

        self.video_frame = ttk.Frame(self.main_frame)
        self.video_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(expand=True)

        ttk.Label(ctrl_frame, text="Camera").pack()
        cam_names = [c[1] for c in self.available_cameras]
        self.cam_var = tk.StringVar(value=cam_names[0])
        self.cam_combo = ttk.Combobox(ctrl_frame, textvariable=self.cam_var, values=cam_names, state='readonly', width=18)
        self.cam_combo.pack(pady=2)
        self.cam_combo.bind('<<ComboboxSelected>>', self.change_camera)

        self.mirror_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl_frame, text="Mirror", variable=self.mirror_var,
                       command=lambda: setattr(config, 'mirror', self.mirror_var.get())).pack(pady=5)

        if VCAM_AVAILABLE:
            self.vcam_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(ctrl_frame, text="Virtual Camera Output", variable=self.vcam_var,
                           command=self.toggle_vcam).pack(pady=5)
        else:
            ttk.Label(ctrl_frame, text="Virtual Camera Unavailable").pack(pady=5)
        
        self.open_fullscreen_cam = ttk.Button(ctrl_frame, text="Open Fullscreen Camera", command=self.open_fullscreen_camera).pack()

        ttk.Label(ctrl_frame, text="Eye Open Threshold (EAR)").pack()
        self.eyes_open_threshold_var = tk.DoubleVar(value=config.ear_threshold)
        self.eyes_open_threshold_scale = ttk.Scale(ctrl_frame, from_=0.05, to=0.3, value=config.ear_threshold, variable=self.eyes_open_threshold_var,
                  command=lambda v: setattr(config, 'ear_threshold', float(v)))
        self.eyes_open_threshold_scale.pack(fill='x', padx=10)

        ttk.Label(ctrl_frame, text="Surprised Threshold (EAR)").pack()
        self.surprised_threshold_var = tk.DoubleVar(value=config.surprised_threshold)
        self.surprised_threshold_scale = ttk.Scale(ctrl_frame, from_=0.25, to=0.5, value=config.surprised_threshold, variable=self.surprised_threshold_var,
                  command=lambda v: setattr(config, 'surprised_threshold', float(v)))
        self.surprised_threshold_scale.pack(fill='x', padx=10)

        ttk.Label(ctrl_frame, text="Mouth Threshold (approx)").pack()
        self.mouth_threshold_var = tk.DoubleVar(value=config.mouth_threshold)
        self.mouth_threshold_scale = ttk.Scale(ctrl_frame, from_=5, to=30, value=config.mouth_threshold, variable=self.mouth_threshold_var,
                  command=lambda v: setattr(config, 'mouth_threshold', float(v)))
        self.mouth_threshold_scale.pack(fill='x', padx=10)

        ttk.Label(ctrl_frame, text="Smile Threshold").pack()
        self.smile_threshold_var = tk.DoubleVar(value=config.smile_threshold)
        self.smile_threshold_scale = ttk.Scale(ctrl_frame, from_=0, to=10, value=config.smile_threshold, variable=self.smile_threshold_var,
                  command=lambda v: setattr(config, 'smile_threshold', float(v)))
        self.smile_threshold_scale.pack(fill='x', padx=10)

        ttk.Label(ctrl_frame, text="Head Box Size").pack()
        self.head_box_var = tk.DoubleVar(value=config.head_padding)
        self.head_box_scale = ttk.Scale(ctrl_frame, from_=0.1, to=0.6, value=config.head_padding, variable=self.head_box_var,
                  command=lambda v: setattr(config, 'head_padding', float(v)))
        self.head_box_scale.pack(fill='x', padx=10)

        color_frame = ttk.Frame(ctrl_frame)
        color_frame.pack(pady=5)
        ttk.Button(color_frame, text="Text Color", command=self.pick_text_color).pack(side='left', padx=2)
        ttk.Button(color_frame, text="Box Color", command=self.pick_box_color).pack(side='left', padx=2)

        ttk.Label(ctrl_frame, text="Special Expression").pack(pady=(15,0))
        self.expr_var = tk.StringVar(value=config.special_mode)
        self.expr_frame = ttk.Frame(ctrl_frame)
        self.expr_frame.pack(fill='x', padx=5)
        expressions = ["AUTO", "silly_tongue", "wink_left", "wink_right", "surprised", "dead", "happy", "sad", "tears"]
        for i, expr in enumerate(expressions):
            ttk.Radiobutton(self.expr_frame, text=expr.replace("_", " "), value=expr, variable=self.expr_var,
                           command=lambda: setattr(config, 'special_mode', self.expr_var.get())).grid(
                               row=i//2, column=i%2, sticky='w', padx=2, pady=2)
        self.refresh_expressions()

        expr_frame = ttk.Frame(ctrl_frame)
        expr_frame.pack(pady=5)
        ttk.Button(expr_frame, text="+ Add", command=self.add_custom_expr, width=8).pack(side='left', padx=2)
        ttk.Button(expr_frame, text="Edit", command=self.edit_custom_expr, width=8).pack(side='left', padx=2)
        ttk.Button(expr_frame, text="Delete", command=self.delete_custom_expr, width=8).pack(side='left', padx=2)
        
        ttk.Button(ctrl_frame, text="Load Background Texture", command=self.load_bg_texture).pack(pady=10)
        ttk.Button(ctrl_frame, text="Load Box Texture", command=self.load_box_texture).pack(pady=10)
        ttk.Button(ctrl_frame, text="Clear Textures", command=self.clear_textures).pack(pady=10)

        settings_frame = ttk.Frame(ctrl_frame)
        settings_frame.pack(pady=5)
        ttk.Button(settings_frame, text="Reset", command=self.reset_config).pack(side='left', padx=2)
        ttk.Button(settings_frame, text="Save", command=self.save_config).pack(side='left', padx=2)
        ttk.Button(settings_frame, text="Load", command=self.load_config).pack(side='left', padx=2)

        ttk.Button(ctrl_frame, text="Quit", command=self.quit).pack(pady=10)
    
    def refresh_expressions(self):
        for widget in self.expr_frame.winfo_children():
            widget.destroy()
        expressions = ["AUTO", "silly_tongue", "wink_left", "wink_right", "surprised", "dead", "happy", "sad", "tears"]
        expressions += [f"custom:{name}" for name in config.custom_expressions.keys()]
        for i, expr in enumerate(expressions):
            display = expr.replace("custom:", "").replace("_", " ")
            ttk.Radiobutton(self.expr_frame, text=display, value=expr, variable=self.expr_var,
                           command=lambda: setattr(config, 'special_mode', self.expr_var.get())).grid(
                               row=i//2, column=i%2, sticky='w', padx=2, pady=2)

    def add_custom_expr(self):
        dialog = CustomExpressionDialog(self.root)
        if dialog.result:
            config.custom_expressions[dialog.result["name"]] = dialog.result["data"]
            self.refresh_expressions()
    
    def open_fullscreen_camera(self):
        if self.fullscreen_cam is None:
            self.fullscreen_dialog = tk.Toplevel(self.root)
            self.fullscreen_dialog.maxsize()
            self.fullscreen_dialog.title("ASCII Face Cover - Fullscreen Cam")
            self.fullscreen_cam = ttk.Label(self.fullscreen_dialog)
            self.fullscreen_cam.pack(expand=True)
            self.fullscreen_dialog.wait_window()
            self.fullscreen_dialog.destroy()
            self.fullscreen_cam = None
    
    def edit_custom_expr(self):
        current = self.expr_var.get()
        if not current.startswith("custom:"):
            messagebox.showinfo("Info", "Select a custom expression to edit")
            return
        name = current.replace("custom:", "")
        data = config.custom_expressions.get(name, {})
        dialog = CustomExpressionDialog(self.root, name, data)
        if dialog.result:
            del config.custom_expressions[name]
            config.custom_expressions[dialog.result["name"]] = dialog.result["data"]
            self.expr_var.set(f"custom:{dialog.result['name']}")
            self.refresh_expressions()

    def delete_custom_expr(self):
        current = self.expr_var.get()
        if not current.startswith("custom:"):
            messagebox.showinfo("Info", "Select a custom expression to delete")
            return
        name = current.replace("custom:", "")
        if messagebox.askyesno("Confirm", f"Delete '{name}'?"):
            del config.custom_expressions[name]
            self.expr_var.set("AUTO")
            config.special_mode = "AUTO"
            self.refresh_expressions()

    def pick_text_color(self):
        color = colorchooser.askcolor(title="Choose Text Color", initialcolor=config.text_color)
        if color[0]:
            # color returned in RGB, convert to BGR tuple
            config.text_color = tuple(int(c) for c in color[0])[::-1]
    
    def pick_box_color(self):
        color = colorchooser.askcolor(title="Choose Box Color", initialcolor=config.box_color)
        if color[0]:
            # color returned in RGB, convert to BGR tuple
            config.box_color = tuple(int(c) for c in color[0])[::-1]

    def reset_config(self, new: dict = defaultConfig.__dict__.copy(), forced: dict = {"mirror": True}, ignore: list = ["virtual_cam_enabled"]):

        self.expr_var.set(forced.get("special_mode" if "special_mode" not in ignore else "", new["special_mode"]))
        if VCAM_AVAILABLE:
            self.vcam_var.set(forced.get("virtual_cam_enabled" if "virtual_cam_enabled" not in ignore else "", new["virtual_cam_enabled"]))
        self.head_box_var.set(forced.get("head_padding" if "head_padding" not in ignore else "", new["head_padding"]))
        self.eyes_open_threshold_var.set(forced.get("ear_threshold" if "ear_threshold" not in ignore else "", new["ear_threshold"]))
        self.mouth_threshold_var.set(forced.get("mouth_threshold" if "mouth_threshold" not in ignore else "", new["mouth_threshold"]))
        self.surprised_threshold_var.set(forced.get("surprised_threshold" if "surprised_threshold" not in ignore else "", new["surprised_threshold"]))
        self.smile_threshold_var.set(forced.get("smile_threshold" if "smile_threshold" not in ignore else "", new["smile_threshold"]))
        self.mirror_var.set(True)

        for key in config.__dict__.keys():
            if key not in ignore:
                setattr(config, key, forced.get(key, new[key]))
        
        self.refresh_expressions()

    def change_camera(self, event=None):
        selected = self.cam_var.get()
        for idx, name in self.available_cameras:
            if name == selected:
                # restart worker on new camera idx
                try:
                    self.worker.open_camera(idx)
                    self.current_cam_idx = idx
                except Exception:
                    pass
                break

    def toggle_vcam(self):
        config.virtual_cam_enabled = getattr(self, 'vcam_var', tk.BooleanVar()).get()
        # virtual cam handling left simple: worker only outputs frames, UI may send to vcam if needed

    def load_bg_texture(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if path:
            config.bg_texture_path = path

    def load_box_texture(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if path:
            config.box_texture_path = path

    def save_config(self):
        filepath = filedialog.asksaveasfilename(filetypes=[("AsciiFaceCover", ".afc")])
        if filepath:
            if not filepath.endswith(".afc"):
                filepath += ".afc"
            config.save(filepath)
            messagebox.showinfo(title="saving status", message=("settings saved successfully"))
    
    def load_config(self):
        filepath = filedialog.askopenfilename(filetypes=[("AsciiFaceCover", ".afc")])
        if filepath:
            success, message = config.load(filepath, self)
            if success:
                self.refresh_expressions()
            messagebox.showinfo(title="loading status", message=("settings loaded successfully" if success else message))

    def clear_textures(self):
        config.bg_texture_path = None
        config.box_texture_path = None

    def update_video(self):
        # fetch latest processed frame from worker
        frame = self.worker.get_frame()
        if frame is not None:
            # Close loading popup on first frame
            if not self.loading_done:
                self.loading_done = True
                self.loading_popup.destroy()
                # Show v4l2loopback warning if needed
                if v4l2_message:
                    messagebox.showwarning("Virtual Camera Setup", v4l2_message)
            h, w = frame.shape[:2]
            rgb_output = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Virtual camera output
            if VCAM_AVAILABLE and config.virtual_cam_enabled:
                try:
                    if self.vcam is None or self.vcam.width != w or self.vcam.height != h:
                        if self.vcam:
                            self.vcam.close()
                        self.vcam = pyvirtualcam.Camera(width=w, height=h, fps=30, fmt=pyvirtualcam.PixelFormat.RGB)
                    self.vcam.send(rgb_output)
                except Exception as e:
                    print(f"Virtual cam error: {e}")
            elif self.vcam and not config.virtual_cam_enabled:
                self.vcam.close()
                self.vcam = None
            
            img = Image.fromarray(rgb_output)
            self.cam_aspect = w / h

            # Fit into UI area
            avail_w = max(100, self.video_frame.winfo_width() - 10)
            avail_h = max(100, self.video_frame.winfo_height() - 10)
            if avail_w / avail_h > self.cam_aspect:
                new_h = avail_h
                new_w = int(avail_h * self.cam_aspect)
            else:
                new_w = avail_w
                new_h = int(avail_w / self.cam_aspect)

            newImg = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=newImg)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            if self.fullscreen_cam:
                avail_w = max(100, self.fullscreen_dialog.winfo_width() - 10)
                avail_h = max(100, self.fullscreen_dialog.winfo_height() - 10)
                if avail_w / avail_h > self.cam_aspect:
                    new_h = avail_h
                    new_w = int(avail_h * self.cam_aspect)
                else:
                    new_w = avail_w
                    new_h = int(avail_w / self.cam_aspect)
                self.fullscreen_dialog.geometry(f"{new_w+10}x{new_h+10}")
                fullImg = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=fullImg)
                self.fullscreen_cam.imgtk = imgtk
                self.fullscreen_cam.configure(image=imgtk)

        # schedule next update based on worker's processing time (adaptive)
        delay_ms = int(max(30, min(100, self.worker.last_process_time * 1000 + 10)))
        self.root.after(delay_ms, self.update_video)

    def quit(self):
        try:
            self.worker.stop()
        except:
            pass
        try:
            if self.vcam:
                self.vcam.close()
        except:
            pass
        self.root.quit()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = AsciiFaceCoverApp()
    app.run()
