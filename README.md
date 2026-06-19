# Clearance

A computer vision pipeline for analyzing traffic from a bicycle-mounted dual-lens VBOX camera, estimating real-world distance between the rider and nearby vehicles using YOLO detection/tracking combined with MiDaS monocular depth estimation.

## Workflow: processing a VBOX camera feed

The VBOX camera records both lenses (left/right) side-by-side in a single frame, with a telemetry strip that needs to be discarded. Processing a new ride video follows three steps:

### Step 1 — Split the raw video into left/right feeds

Open `utils/split_video.py` and set the input path and output path relative to the project root:

```python
input_video = './videos/d2.mp4'
output_dir = './videos/'
```

Revise the split coordinates if needed:

```python
TOP, BOTTOM = 0, 540
LEFT1, RIGHT1 = 0, 960
LEFT2, RIGHT2 = 960, 1920
```

Run it:

```bash
python utils/split_video.py
```

This crops out the telemetry strip and splits the dual-lens frame into two separate videos: `<name>_left.mp4` and `<name>_right.mp4`.



### Step 2 — Run the pipeline on the split video

```bash
python pipeline.py path/to/video_left.mp4
```

This will:
- Detect vehicles, pedestrians, and e-rickshaws (YOLO26n COCO model + fine-tuned e-rickshaw model)
- Track each object across frames
- Estimate a depth map per frame using MiDaS (`DPT_Hybrid`)
- Convert each object's depth reading into a real-world distance (cm) using the calibrated rational curve {y = a / (x + b) + c}
- Suppress rider-on-vehicle double detections (e.g. a person on a bicycle/e-rickshaw)

Outputs are written to `detection_output/` by default:
- `<video_name>_<timestamp>.mp4` — annotated video with bounding boxes, object IDs, and distances
- `<video_name>_<timestamp>.csv` — per-frame, per-object detection log (class, confidence, box coordinates, distance)

## Recalibrating for a different camera

The distance conversion in `pipeline.py` (`VideoProcessor.distance_levit`) uses fixed coefficients fitted to the current VBOX camera's depth output:

```python
a, b, c = 157994.966563, -372.880209, -13.939203
return (a / (median_depth + b)) + c
```

These coefficients are camera-specific — they depend on the lens, sensor, and mounting position. **If you switch to a different camera, you need to recalibrate:**

1. Capture a calibration image with the new camera — a frame containing reference points at known real-world distances (e.g. floor tiles, road markings) works well.
2. Open `utils/calibrate_rational.py` and update:
   ```python
   CALIBRATION_IMAGE_PATH = "../videos/stripes.png"
   ```
   and the `CALIBRATION_POINTS` list, with each entry as `(pixel_x, pixel_y, real_world_distance_cm)` for a point you can identify in the calibration image.
3. Run it:
   ```bash
   python utils/calibrate_rational.py
   ```
   This runs MiDaS on the calibration image, fits a rational curve (`y = a / (x + b) + c`) between raw depth output and your known distances, and prints the new `a, b, c` coefficients along with a plot to visually verify the fit.
4. Copy the printed coefficients into `distance_levit()` in `pipeline.py`, replacing the existing values.

`calibrate_dpt_cubic.py` is an alternative calibration script that fits a cubic curve instead of a rational one, kept for comparison — the rational fit is the one currently used in the pipeline.
