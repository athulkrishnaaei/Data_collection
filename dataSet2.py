import os
import time
import argparse
import glob
import cv2
import torch
from datetime import datetime
from pathlib import Path

# ───── USER CONFIG ─────
SERIAL_ENABLED = False
EVAL_MODE = False
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

# ─── ZONE RECTANGLES IN CM ───
ZONE_RECTS_CM = [
    (0.0, 8.0, 0.0, 10.0),
    (8.0, 16.0, 0.0, 10.0),
    (16.0, 24.0, 0.0, 10.0),
    (24.0, 40.0, 0.0, 10.0),
]
ZONE_RECTS_PX = [
    (int(xs * PX_PER_CM), int(xe * PX_PER_CM),
     int(ys * PX_PER_CM), int(ye * PX_PER_CM))
    for xs, xe, ys, ye in ZONE_RECTS_CM
]

# ───── CONSTANTS ─────
STRIP_H_PX = int(DETECT_HEIGHT_CM * PX_PER_CM)
OFF_PY_PX = int(DETECT_Y_OFFSET_CM * PX_PER_CM)
ZONE_COUNT = len(ZONE_RECTS_PX)

# ───── ARG PARSING ─────
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='path to .pt file')
parser.add_argument('--source', required=True, help='file/folder/usbN/URL')
parser.add_argument('--normal_thresh', type=float, default=0.70, help='conf thresh for normal apples')
parser.add_argument('--rotten_thresh', type=float, default=0.85, help='conf thresh for rotten apples')
parser.add_argument('--rotate', type=int, choices=[0, 90, 180, 270], default=0,
                    help='rotate incoming frames to correct mounting')
parser.add_argument('--resolution', default=None, help='WxH for camera')
parser.add_argument('-H', '--headless', action='store_true', help='disable GUI')
parser.add_argument('-q', '--quiet', action='store_true', help='suppress prints')
args = parser.parse_args()

NORMAL_THRESH = args.normal_thresh
ROTTEN_THRESH = args.rotten_thresh

# ───── ROTATION MAP ─────
rot_map = {
    0: lambda z: z,
    90: lambda z: (z + 1) % ZONE_COUNT,
    180: lambda z: z,
    270: lambda z: (z + ZONE_COUNT - 1) % ZONE_COUNT,
}

# ───── SERIAL INIT ─────
try:
    import serial
    arduino = None
    if SERIAL_ENABLED:
        arduino = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.01)
        time.sleep(2)
        if not args.quiet:
            print('[INFO] Serial ENABLED')
        arduino.write(b'E1')
    else:
        if not args.quiet:
            print('[INFO] Serial DISABLED')
except Exception as e:
    if not args.quiet:
        print(f'[WARN] Serial init failed: {e}')
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
        y0 = max(0, min(OFF_PY_PX, h - STRIP_H_PX))
        y1 = y0 + STRIP_H_PX
        roi = frame[y0:y1]

        # 5) Draw zones
        display_frame = frame.copy()
        if not args.headless:
            cv2.rectangle(display_frame, (0, y0), (w, y1), (255, 255, 0), LINE_THICKNESS)
            for xs, xe, ys, ye in ZONE_RECTS_PX:
                cv2.rectangle(display_frame, (xs, y0 + ys), (xe, y0 + ye), (255, 255, 0), LINE_THICKNESS)

        # 6) Inference
        model.conf = min(NORMAL_THRESH, ROTTEN_THRESH)
        results = model(roi[..., ::-1])  # BGR->RGB
        dets = results.xyxy[0]

        # 7) Process detections
        zones = ['0'] * ZONE_COUNT
        for *xyxy, conf, cls_i in dets.cpu().numpy():
            x1, y1_, x2, y2_ = map(int, xyxy)
            lbl = labels[int(cls_i)]
            thr = ROTTEN_THRESH if 'rotten' in lbl.lower() else NORMAL_THRESH
            if conf < thr:
                continue
            cx, cy = (x1 + x2) // 2, (y1_ + y2_) // 2
            orig = next((i for i, (xs, xe, ys, ye) in enumerate(ZONE_RECTS_PX)
                         if xs <= cx < xe and ys <= cy < ye), None)
            if orig is None:
                continue
            zi = rot_map[args.rotate](orig)
            zones[zi] = '1'

            # Save as .txt file in a unified folder (`detected_images`)
            if TAKE_PHOTO_ENABLED:
                out_dir = os.path.join(PHOTO_BASE_DIR, DETECTED_IMAGES_DIR)
                label_dir = os.path.join(out_dir, 'labels')
                os.makedirs(label_dir, exist_ok=True)
                crop = roi[y1_:y2_, x1:x2]
                pct = int(conf * 100)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                fname = f"{lbl}_{pct}%_{ts}.jpg"
                cv2.imwrite(os.path.join(out_dir, fname), crop)

                # Save the .txt label file with bounding box details
                txt_filename = os.path.join(label_dir, fname.replace('.jpg', '.txt'))
                with open(txt_filename, 'w') as f:
                    # Format: class_id x_center y_center width height confidence
                    x_center = (x1 + x2) / 2 / w
                    y_center = (y1_ + y2_) / 2 / h
                    width = (x2 - x1) / w
                    height = (y2_ - y1_) / h
                    f.write(f"{int(cls_i)} {x_center} {y_center} {width} {height} {conf}\n")

                if not args.quiet:
                    print(f"[INFO] Saved {fname} and its corresponding label file to {label_dir}")

        # 8) Output zones (optional serial communication)
        # msg = ''.join(f"{chr(65 + i)}{zones[i]}" for i in range(ZONE_COUNT))
        # if SERIAL_ENABLED and arduino and msg != prev_msg:
        #     arduino.write((msg + '\n').
                # if SERIAL_ENABLED and arduino and msg != prev_msg:
        #     arduino.write((msg + '\n').encode())
        #     arduino.flush()
        #     prev_msg = msg
        # if not args.quiet:
        #     print(msg)

        # 9) Display & FPS
        dt = time.perf_counter() - t0
        fps_buf.append(1 / dt)
        if len(fps_buf) > 200:
            fps_buf.pop(0)
        if not args.headless:
            fps = sum(fps_buf) / len(fps_buf)
            cv2.putText(display_frame, f"FPS:{fps:.1f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow('Detect', display_frame)
            if cv2.waitKey(1) == ord('q'):
                break

except KeyboardInterrupt:
    pass

# ───── CLEANUP ─────
if src_type in ('video', 'usb', 'ip'):
    cap.release()
if not args.headless:
    cv2.destroyAllWindows()
if SERIAL_ENABLED and arduino:
    arduino.close()
