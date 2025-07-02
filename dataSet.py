import os
import time
import argparse
import glob
import cv2
import torch
from datetime import datetime

# ───── USER CONFIG ─────
SERIAL_ENABLED = False
EVAL_MODE = False
TAKE_PHOTO_ENABLED = True  # enable capture
PHOTO_BASE_DIR = 'captures'  # base folder
NORMAL_DIR_NAME = 'normal'
ROTTEN_DIR_NAME = 'rotten'

SERIAL_PORT = '/dev/ttyUSB0'
SERIAL_BAUD = 115200

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

# ───── VIDEO/IMAGE SOURCE ─────
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

        # 3) Resize if needed
        if args.resolution and src_type == 'video':
            frame = cv2.resize(frame, (W, H))

        display_frame = frame.copy()

        # 4) Inference on full frame
        results = model(frame[..., ::-1])  # BGR->RGB
        dets = results.xyxy[0].cpu().numpy()

        # 5) Process & draw detections, save full frame
        for *xyxy, conf, cls_i in dets:
            x1, y1_, x2, y2_ = map(int, xyxy)
            lbl = labels[int(cls_i)]
            thr = ROTTEN_THRESH if 'rotten' in lbl.lower() else NORMAL_THRESH
            if conf < thr:
                continue

            # draw bounding box + label on display
            cv2.rectangle(display_frame, (x1, y1_), (x2, y2_), (0, 255, 0), 2)
            cv2.putText(display_frame,
                        f"{lbl} {conf:.2f}",
                        (x1, y1_ - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

            # save full frame (no boxes) + label file
            if TAKE_PHOTO_ENABLED:
                cat = ROTTEN_DIR_NAME if 'rotten' in lbl.lower() else NORMAL_DIR_NAME
                out_dir = os.path.join(PHOTO_BASE_DIR, cat)
                label_dir = os.path.join(out_dir, 'labels')
                os.makedirs(label_dir, exist_ok=True)

                pct = int(conf * 100)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                fname = f"{lbl}_{pct}%_{ts}.jpg"
                cv2.imwrite(os.path.join(out_dir, fname), frame)

                # write YOLO-style .txt
                txt_path = os.path.join(label_dir, fname.replace('.jpg', '.txt'))
                with open(txt_path, 'w') as f:
                    h_f, w_f = frame.shape[:2]
                    x_center = ((x1 + x2) / 2) / w_f
                    y_center = ((y1_ + y2_) / 2) / h_f
                    width = (x2 - x1) / w_f
                    height = (y2_ - y1_) / h_f
                    f.write(f"{int(cls_i)} {x_center} {y_center} {width} {height} {conf}\n")

                if not args.quiet:
                    print(f"[INFO] Saved full frame {fname} (+ label) to {label_dir}")

        # 6) Display & FPS
        dt = time.perf_counter() - t0
        fps_buf.append(1 / dt)
        if len(fps_buf) > 200:
            fps_buf.pop(0)
        if not args.headless:
            fps = sum(fps_buf) / len(fps_buf)
            cv2.putText(display_frame, f"FPS:{fps:.1f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow('Detect', display_frame)
            if cv2.waitKey(1) == ord('q'):
                break

except KeyboardInterrupt:
    pass

# ───── CLEANUP ─────
if src_type == 'video':
    cap.release()
if not args.headless:
    cv2.destroyAllWindows()
if SERIAL_ENABLED and 'arduino' in locals() and arduino:
    arduino.close()
