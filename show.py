import os
import glob
import cv2

# ───── USER CONFIG ─────
BASE_DIR      = 'captures'
CATEGORIES    = ('normal', 'rotten')
DISPLAY_SCALE = 2.0   # change to e.g. 1.5, 2.0, etc.

# map class IDs in your .txt files to human-readable names
LABELS = {
    0: 'normal',
    1: 'rotten',
}

def draw_boxes_from_labels(image, label_path, color=(0,255,0), thickness=2):
    """
    Reads a YOLO-style .txt and draws boxes & label name onto `image`.
    Each line in .txt: class_id x_center y_center width height confidence
    (all normalized 0–1).
    """
    h, w = image.shape[:2]
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            x_c, y_c, bw, bh = map(float, parts[1:5])
            # convert normalized coords back to pixel values
            x1 = int((x_c - bw/2) * w)
            y1 = int((y_c - bh/2) * h)
            x2 = int((x_c + bw/2) * w)
            y2 = int((y_c + bh/2) * h)

            # draw the box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            # draw the label name just above the top-left corner of the box
            label = LABELS.get(cls_id, str(cls_id))
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

def display_labeled_images(base_dir=BASE_DIR,
                           categories=CATEGORIES,
                           scale=DISPLAY_SCALE):
    for cat in categories:
        img_paths = sorted(glob.glob(os.path.join(base_dir, cat, '*.jpg')))
        label_dir = os.path.join(base_dir, cat, 'labels')

        for img_path in img_paths:
            fname = os.path.basename(img_path)
            txt_path = os.path.join(label_dir, fname.replace('.jpg', '.txt'))

            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Can't load {img_path}")
                continue

            # overlay boxes + label names if .txt exists
            if os.path.isfile(txt_path):
                draw_boxes_from_labels(img, txt_path)
            else:
                print(f"[WARN] No label file for {fname}")

            # scale up for display
            disp = cv2.resize(
                img, None,
                fx=scale, fy=scale,
                interpolation=cv2.INTER_LINEAR
            )

            # make window resizable & show
            window = f"{cat.upper()} – {fname}"
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.imshow(window, disp)

            key = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow(window)
            if key == ord('q'):
                return

if __name__ == '__main__':
    display_labeled_images()
