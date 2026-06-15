# %%
# @title 1. Import Required Libraries

import random
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import cv2
from ultralytics import YOLO
import numpy as np

# %%
# @title 2. Configuration & Paths

INPUT_VIDEO = "videos\\d14_left.mp4"
OUTPUT_DIR = "detection_output\\d14_left"
ERICK_MODEL_PATH = "best.pt"


# Detection settings
CONF_THRESHOLD = 0.40  # minimum confidence to show a detection

PERSON_VEHICLE_IOU_THRESHOLD = 0.20
PERSON_VEHICLE_IOA_THRESHOLD = 0.80
PERSON_VEHICLE_CENTROID_DIST_THRESHOLD = 30
VEHICLE_CLASSES = {'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'erickshaw'}

COCO_CLASSES_KEEP = {
    0:  'person',
    1:  'bicycle',
    2:  'car',
    3:  'motorcycle',
    5:  'bus',
    7:  'truck',
}

# %%
# @title 3. Class Colors & Model Setup

# Box colors per class (BGR format)
CLASS_COLORS = {
    'erickshaw' : (0,   255, 0),    # green
    'person'     : (255, 178, 50),   # blue
    'bicycle'    : (0,   165, 255),  # orange
    'car'        : (0,   0,   255),  # red
    'motorcycle' : (255, 0,   255),  # magenta
    'bus'        : (255, 255, 0),    # cyan
    'truck'      : (128, 0,   128),  # purple
}

# Load YOLO model
coco_model = YOLO("yolov8m.pt")
erick_model = YOLO(ERICK_MODEL_PATH)

# %%
# @title 4. Initialize Output Files & Video Writer

output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

video_stem = Path(INPUT_VIDEO).stem
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

out_path = output_dir / f"{video_stem}_tracking_{timestamp}.mp4"
csv_path = output_dir / f"{video_stem}_tracking_{timestamp}.csv"

# Video input/output setup
cap = cv2.VideoCapture(INPUT_VIDEO)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration_sec = total_frames / video_fps

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(out_path), fourcc, video_fps, (width, height))

# CSV writer
csv_file = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'frame', 'timestamp_sec', 'camera',
    'class', 'object', 'confidence',
    'x_center', 'y_center', 'width', 'height',
    'x1', 'y1', 'x2', 'y2', 'distance'
])

total_detections = defaultdict(int)

# %%
# Pixel coordinates from your image
pts_px = np.array([

    [567, 495], [488, 496], [647, 491], 
    [496, 474], [564, 473], [633, 470], # tiles
    [389, 367], [410, 367], [455, 366], [481, 364], [527, 361], [550, 360], [597, 357], [621, 354], [670, 350],
    [427, 486], [509, 480], [679, 476], [760, 464], [296, 479],
    [625, 395], [536, 400], [492, 401], [797, 376], [863, 366], [895, 361], [413, 403], [373, 404], [313, 403], [289, 399],
], dtype=np.float32)

# Real-world coordinates in meters (relative to your bike)
pts_cm = np.array([
    [0, 163], [-30, 163], [30, 163], 
    [-30, 193], [0, 193], [30, 193], 
    [-212, 590], [-182, 590], [-121, 590], [-91, 590], [-32, 590], [-2, 590], [63, 590], [93, 590], [153, 590],
    [-32, 150], [-2, 150], [63, 150], [93, 150], [-93, 150],
    [63, 300], [-2, 300], [-32, 300], [182, 300], [243, 300], [273, 300], [-93, 300], [-123, 300], [-184, 300], [-214, 300],

], dtype=np.float32)

# Calculate the Homography matrix
H, mask = cv2.findHomography(pts_px, pts_cm)

def distance(point: tuple) -> float:
    return cv2.perspectiveTransform(
        np.array([[point]], dtype=np.float32),
        H
    )[0][0][1]

# %%
# @title 5. Helper Functions

def compute_iou(boxA, boxB):
    """
    Compute Intersection over Union between two boxes.
    Each box is (x1, y1, x2, y2).
    Returns IoU value between 0 and 1.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h

    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter

    return inter / union if union > 0 else 0.0


def compute_ioa(boxA, boxB):
    """
    Compute Intersection over Area between two boxes.
    Each box is (x1, y1, x2, y2).
    Returns IoA value between 0 and 1.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h

    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    areaMin = min(areaA, areaB)

    return inter / areaMin if areaMin > 0 else 0.0


def compute_centroid(box):
    """Compute centroid of a box (x1, y1, x2, y2)."""
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return (cx, cy)


def draw_box(frame, x1, y1, x2, y2, dist, label, conf, color):
    """Draw a bounding box with label on frame."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f} {dist:.1f}cm"
    txt_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    
    # Background rectangle for text
    cv2.rectangle(frame,
                  (x1, y1 - txt_size[1] - 6),
                  (x1 + txt_size[0] + 2, y1),
                  color, -1)
    cv2.putText(frame, text,
                (x1 + 1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    return frame


def is_bike_rider(pbox, vbox):
    """
    Determine if a person box is a rider on a vehicle.
    Uses IoU, IoA, and centroid distance heuristics.
    """
    if (compute_iou(pbox, vbox) >= PERSON_VEHICLE_IOU_THRESHOLD or 
        compute_ioa(pbox, vbox) >= PERSON_VEHICLE_IOA_THRESHOLD):
        return True
    
    person_center_x = (pbox[0] + pbox[2]) / 2
    person_center_y = (pbox[1] + pbox[3]) / 2
    vehicle_center_x = (vbox[0] + vbox[2]) / 2
    vehicle_center_y = (vbox[1] + vbox[3]) / 2

    # Calculate Euclidean distance between centroids
    dist = ((person_center_x - vehicle_center_x) ** 2 +
                (person_center_y - vehicle_center_y) ** 2) ** 0.5

    return dist < PERSON_VEHICLE_CENTROID_DIST_THRESHOLD


def suppress_detections(all_dets):
    """
    Remove 'person' detections whose bounding boxes overlap
    significantly with vehicle boxes (i.e., riders on vehicles).
    
    Input:  list of dicts with keys: label, conf, x1, y1, x2, y2
    Output: (filtered_list, suppressed_count)
    """
    vehicle_boxes = [
        (d['x1'], d['y1'], d['x2'], d['y2'])
        for d in all_dets
        if d['label'] in VEHICLE_CLASSES
    ]

    erick_boxes = [
        (d['x1'], d['y1'], d['x2'], d['y2'])
        for d in all_dets
        if d['label'] == 'erickshaw'
    ]

    if not vehicle_boxes:
        return all_dets, 0  # no vehicles → keep all persons

    filtered = []
    suppressed_count = 0

    for det in all_dets:
        if det['label'] == 'person':
            person_box = (det['x1'], det['y1'], det['x2'], det['y2'])
            # Check overlap with every vehicle box
            is_rider = any(
                is_bike_rider(person_box, vbox)
                for vbox in vehicle_boxes
            )
            if is_rider:
                suppressed_count += 1
                continue  # skip this person — they're a rider

        if det['label'] in VEHICLE_CLASSES:
            vehicle_box = (det['x1'], det['y1'], det['x2'], det['y2'])
            overlaps_erick = any(
                compute_iou(vehicle_box, ebox) >= PERSON_VEHICLE_IOU_THRESHOLD
                for ebox in erick_boxes
            )
            if overlaps_erick:
                suppressed_count += 1
                continue

        filtered.append(det)

    return filtered, suppressed_count

# %%
# @title 6. Main Detection & Tracking Function

def run_tracking(crop, frame_idx, timestamp, camera_name):
    """
    Run YOLO tracking on a frame crop.
    
    Steps:
    1. Collect all detections from YOLO
    2. Suppress person boxes overlapping with vehicles (riders)
    3. Draw bounding boxes and prepare CSV rows
    
    Returns:
        annotated: Frame with drawn detections
        csv_rows: List of detection rows for CSV
    """
    annotated = crop.copy()
    all_dets = []

    # Only recognises Erick
    er_results = erick_model.track(crop, conf=CONF_THRESHOLD, verbose=False, persist=True)[0]
    if er_results.boxes.id is not None:
        boxes = er_results.boxes
        track_ids = er_results.boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            cls_id = int(box.cls[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            all_dets.append({
                'tracking_id': track_id,
                'object': 'erickshaw' + str(track_id),
                'label' : 'erickshaw',
                'conf'  : float(box.conf[0]),
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'distance': distance(((x1 + x2) / 2, y2))
            })

    # COCO model (persons, vehicles, etc.)
    co_results = coco_model.track(crop, conf=CONF_THRESHOLD, verbose=False, persist=True)[0]

    if co_results.boxes.id is not None:
        boxes = co_results.boxes
        track_ids = co_results.boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            cls_id = int(box.cls[0])
            if cls_id not in COCO_CLASSES_KEEP:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            all_dets.append({
                'tracking_id': track_id,
                'object': COCO_CLASSES_KEEP[cls_id] + str(track_id),
                'label': COCO_CLASSES_KEEP[cls_id],
                'conf': float(box.conf[0]),
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'distance': distance(((x1 + x2) / 2, y2))
            })

    # Suppress persons on vehicles (riders)
    filtered_dets, n_suppressed = suppress_detections(all_dets)
    total_detections['_suppressed'] += n_suppressed

    # Draw boxes and build CSV rows
    csv_rows = []
    for det in filtered_dets:
        label = det['label']
        obj = det['object']
        conf = det['conf']
        dist = det['distance']
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        
        # Random color per tracking ID for consistency
        random.seed(det['tracking_id'])
        color = (random.randint(0, 255), random.randint(0, 255), 
                 random.randint(0, 255))

        annotated = draw_box(annotated, x1, y1, x2, y2, dist, obj, conf, color)

        # Normalize coordinates
        cx = (x1 + x2) / 2 / crop.shape[1]
        cy = (y1 + y2) / 2 / crop.shape[0]
        bw = (x2 - x1) / crop.shape[1]
        bh = (y2 - y1) / crop.shape[0]
        
        csv_rows.append([frame_idx, timestamp, camera_name,
                         label, obj, f"{conf:.3f}",
                         f"{cx:.4f}", f"{cy:.4f}",
                         f"{bw:.4f}", f"{bh:.4f}",
                         x1, y1, x2, y2, dist])
        total_detections[label] += 1

    return annotated, csv_rows

# %%
# @title 7. Main Processing Loop

frame_idx = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        
        timestamp = frame_idx / video_fps

        # Run detection and tracking
        ann_frame, left_rows = run_tracking(
            frame, frame_idx, timestamp, 'camera')

        # Write to CSV
        for row in left_rows:
            csv_writer.writerow(row)

        # Add frame info overlay
        cv2.putText(ann_frame,
                    f"Frame: {frame_idx}  T: {timestamp:.1f}s",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

        # Write annotated frame to output video
        out.write(ann_frame)

        # Progress indicator
        pct = frame_idx / total_frames * 100
        print(f"\rProgress: {pct:.1f}%", end="")

except KeyboardInterrupt:
    print("\nProcessing interrupted by user.")
finally:
    # Cleanup
    cap.release()
    out.release()
    csv_file.close()

# %%
results = coco_model(source="videos\d14_f007260.jpg",
                              conf=CONF_THRESHOLD, verbose=False)


# %%
all_dets = []

for result in results:
    if result.boxes is None:
        continue
    for box in result.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        all_dets.append({
            'label' : result.names[cls_id],
            'conf'  : float(box.conf[0]),
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        })
    annotated = result.plot()
    cv2.imwrite("videos\\annotated.jpg", annotated)

filtered_dets, n_suppressed = suppress_detections(all_dets)
print(n_suppressed)

# %%
img_l = []
for det in filtered_dets:
    label = det['label']
    conf  = det['conf']
    x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
    color = CLASS_COLORS.get(label, (200, 200, 200))
    print(label)



