import os
import time
import argparse
import glob
import cv2
import torch
from datetime import datetime

# ───── USER CONFIG ─────
SERIAL_ENABLED = False
TAKE_PHOTO_ENABLED = True  # enable capture
PHOTO_BASE_DIR = 'capture'  # base folder
DETECTED_IMAGES_DIR = 'detected_images'  # Unified folder for all detected images and labels

SERIAL_PORT = '/dev/ttyUSB0'
SERIAL_BAUD = 115200

DETECT_HEIGHT_CM = 10.0
DETECT_Y_OFFSET_CM = 8.5
PX_PER_CM = 20
LINE_THICKNESS = 3
ZOOM_FACTOR = 1.0

# ───── CONSTANTS ─────
STRIP_H_PX = int(DETECT_HEIGHT_CM * PX_PER_CM)
OFF_PY_PX = int(DETECT_Y_OFFSET_CM * PX_PER_CM)

# Add this variable to control bounding box display in GUI
SHOW_BBOX_IN_GUI = True  # Set to False if you do not want to show bounding boxes in GUI

# ───── ARG PARSING ─────
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='path to .pt file')
parser.add_argument('--source', required=True, help='file/folder/usbN/URL')
parser.add_argument('--normal_thresh', type=float, default=0.70, help='conf thresh for normal apples')
parser.add_argument('--rotten_thresh', type=float, default=0.85, help='conf thresh for rotten apples')
parser.add_argument('--rotate', type=int, choices=[0, 90, 180, 270], default=0,
                    help='rotate incoming frames to correct mounting')
parser.add_argument('--resolution', default=None, help='WxH for camera')
args = parser.parse_args()

NORMAL_THRESH = args.normal_thresh
ROTTEN_THRESH = args.rotten_thresh

# ───── ROTATION MAP ─────
rot_map = {
    0: lambda z: z,
    90: lambda z: (z + 1) % 4,
    180: lambda z: z,
    270: lambda z: (z + 3) % 4,
}

# ───── SERIAL INIT ─────
try:
    import serial
    arduino = None
    if SERIAL_ENABLED:
        arduino = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.01)
        time.sleep(2)
        arduino.write(b'E1')
except Exception as e:
    SERIAL_ENABLED = False

# ───── MODEL LOAD (YOLOv5 via torch.hub) ─────
model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.model, force_reload=False)
model.conf = 0.25  # base confidence, overridden per-run
labels = model.names

# ───── VIDEO SOURCE ─────
img_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
if os.path.isdir(args.source):
    imgs = sorted(glob.glob(os.path.join(args.source, '*')))
    src_type = 'folder'
elif os.path.isfile(args.source) and os.path.splitext(args.source)[1].lower() in img_ext:
    imgs = [args.source]
    src_type = 'image'
else:
    src_type = 'video'
    cam = int(args.source[3:]) if args.source.startswith('usb') else args.source
    cap = cv2.VideoCapture(cam)
    if args.resolution:
        W, H = map(int, args.resolution.split('x'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

prev_msg = None
fps_buf = []
idx = 0

try:
    while True:
        t0 = time.perf_counter()

        # 1) Grab frame
        if src_type in ('image', 'folder'):
            if idx >= len(imgs):
                break
            frame = cv2.imread(imgs[idx])
            idx += 1
        else:
            ret, frame = cap.read()
            if not ret:
                break

        # 2) Rotate if needed
        if args.rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif args.rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif args.rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 3) Resize & Zoom
        if args.resolution and src_type == 'video':
            frame = cv2.resize(frame, (W, H))
        h, w = frame.shape[:2]
        if ZOOM_FACTOR > 1.0:
            nw, nh = int(w / ZOOM_FACTOR), int(h / ZOOM_FACTOR)
            x0, y0 = (w - nw) // 2, (h - nh) // 2
            frame = cv2.resize(frame[y0:y0 + nh, x0:x0 + nw], (w, h))

        # 4) Compute ROI
        roi = frame  # Take the full frame as ROI for apple detection

        # 5) Inference
        model.conf = min(NORMAL_THRESH, ROTTEN_THRESH)
        results = model(roi[..., ::-1])  # BGR->RGB
        dets = results.xyxy[0]

        # 6) Process detections
        for *xyxy, conf, cls_i in dets.cpu().numpy():
            x1, y1_, x2, y2_ = map(int, xyxy)
            lbl = labels[int(cls_i)]
            thr = ROTTEN_THRESH if 'rotten' in lbl.lower() else NORMAL_THRESH
            if conf < thr:
                continue

            # Save the full image (no cropping) without bounding boxes
            if TAKE_PHOTO_ENABLED:
                out_dir = os.path.join(PHOTO_BASE_DIR, DETECTED_IMAGES_DIR)
                label_dir = os.path.join(out_dir, 'labels')
                os.makedirs(label_dir, exist_ok=True)

                # Save the full frame without cropping
                pct = int(conf * 100)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                fname = f"{lbl}_{pct}%_{ts}.jpg"
                cv2.imwrite(os.path.join(out_dir, fname), frame)  # Save full frame

                # Save the .txt label file with bounding box details
                txt_filename = os.path.join(label_dir, fname.replace('.jpg', '.txt'))
                with open(txt_filename, 'w') as f:
                    # Format: class_id x_center y_center width height confidence
                    x_center = (x1 + x2) / 2 / w
                    y_center = (y1_ + y2_) / 2 / h
                    width = (x2 - x1) / w
                    height = (y2_ - y1_) / h
                    f.write(f"{int(cls_i)} {x_center} {y_center} {width} {height} {conf}\n")

            # Draw bounding boxes only for display (not saved image)
            if SHOW_BBOX_IN_GUI:  # Only draw if the variable is True
                cv2.rectangle(frame, (x1, y1_), (x2, y2_), (0, 255, 0), 2)
                cv2.putText(frame, f'{lbl} {conf:.2f}', (x1, y1_ - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 7) Display & FPS
        dt = time.perf_counter() - t0
        fps_buf.append(1 / dt)
        if len(fps_buf) > 200:
            fps_buf.pop(0)
        fps = sum(fps_buf) / len(fps_buf)
        cv2.putText(frame, f"FPS:{fps:.1f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow('Detect', frame)  # Display the frame with bounding boxes
        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    pass

# ───── CLEANUP ─────
if src_type in ('video', 'usb', 'ip'):
    cap.release()
cv2.destroyAllWindows()
if SERIAL_ENABLED and arduino:
    arduino.close()
