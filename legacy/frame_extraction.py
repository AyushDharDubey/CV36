import cv2
import os
from pathlib import Path

# ============================================================
#  CONFIGURATION — Edit these paths before running
# ============================================================

VIDEO_FOLDER   = r"D:\video_data_analysis\vbox_videos"
OUTPUT_FOLDER  = r"D:\video_data_analysis\frames_extracted"
EXTRACT_FPS    = 1    # 0.5 = 1 frame every 1 seconds
MIN_BLUR_SCORE = 100    # blur filter threshold (higher = stricter)
IMG_FORMAT     = "jpg"  # jpg or png

# ── VideoVBOX HD layout (1920×1080) ─────────────────────────
# Top half (0→540)   : both cameras side by side
# Bottom half(540→1080): green telemetry overlay — discarded
# Left  camera : x=0    to x=960,  y=0 to y=540
# Right camera : x=960  to x=1920, y=0 to y=540

FRAME_W, FRAME_H = 1920, 1080

LEFT_CROP  = (0,   0, 960,  540)   # (x1, y1, x2, y2)
RIGHT_CROP = (960, 0, 1920, 540)

# ============================================================

SUPPORTED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.MOV', '.AVI'}


def is_blurry(frame, threshold):
    """Returns True if frame is too blurry to be useful."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < threshold, score


def crop_frame(frame, crop):
    """Crop a frame given (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = crop
    return frame[y1:y2, x1:x2]


def verify_frame_size(cap, video_name):
    """Check that video resolution matches expected 1920x1080."""
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w != FRAME_W or h != FRAME_H:
        print(f"  [WARNING] {video_name} is {w}x{h}, expected {FRAME_W}x{FRAME_H}")
        print(f"  [WARNING] Crop coordinates may be wrong — check output visually!")
        return False
    return True


def extract_frames_from_video(video_path, out_left_dir, out_right_dir,
                               fps_target, blur_threshold):
    """
    Extract frames from a VideoVBOX HD video.
    Splits each frame into left and right camera crops.
    Skips blurry frames.
    Returns (extracted, skipped_blur, total_checked)
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"  [ERROR] Could not open: {video_path.name}")
        return 0, 0, 0

    verify_frame_size(cap, video_path.name)

    video_fps     = cap.get(cv2.CAP_PROP_FPS)
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec  = total_frames / video_fps if video_fps > 0 else 0
    frame_interval = max(1, int(video_fps / fps_target))

    print(f"  Resolution    : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"  Video FPS     : {video_fps:.1f}")
    print(f"  Duration      : {duration_sec/60:.1f} min  ({total_frames:,} frames)")
    print(f"  Extracting    : every {frame_interval} frames  (~{fps_target} fps)")

    os.makedirs(out_left_dir,  exist_ok=True)
    os.makedirs(out_right_dir, exist_ok=True)

    frame_idx       = 0
    extracted       = 0
    skipped_blur    = 0
    total_checked   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            total_checked += 1

            # Crop left camera and check blur
            left_crop  = crop_frame(frame, LEFT_CROP)
            blurry, score = is_blurry(left_crop, blur_threshold)

            if blurry:
                skipped_blur += 1
                frame_idx += 1
                continue

            # Crop right camera
            right_crop = crop_frame(frame, RIGHT_CROP)

            # Save both crops
            stem = video_path.stem
            fname = f"{stem}_f{frame_idx:06d}.{IMG_FORMAT}"

            cv2.imwrite(os.path.join(out_left_dir,  fname), left_crop)
            cv2.imwrite(os.path.join(out_right_dir, fname), right_crop)
            extracted += 1

        frame_idx += 1

    cap.release()
    return extracted, skipped_blur, total_checked


def main():
    video_folder  = Path(VIDEO_FOLDER)
    output_folder = Path(OUTPUT_FOLDER)

    video_files = sorted([
        f for f in video_folder.iterdir()
        if f.suffix in SUPPORTED_EXTENSIONS
    ])

    if not video_files:
        print(f"\n[!] No video files found in: {video_folder}")
        print(f"    Supported: {SUPPORTED_EXTENSIONS}")
        return

    print("=" * 65)
    print("FRAME EXTRACTOR — VideoVBOX HD Dual Camera")
    print("=" * 65)
    print(f"  Videos found    : {len(video_files)}")
    print(f"  Output folder   : {output_folder}")
    print(f"  Extract rate    : {EXTRACT_FPS} fps (1 frame every {1/EXTRACT_FPS:.0f}s)")
    print(f"  Blur threshold  : {MIN_BLUR_SCORE}")
    print(f"  Left  crop      : {LEFT_CROP}  → 960×540 px")
    print(f"  Right crop      : {RIGHT_CROP} → 960×540 px")
    print("=" * 65)

    grand_extracted  = 0
    grand_skipped    = 0
    grand_checked    = 0

    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] {video_path.name}")

        # Separate subfolders per video per camera
        out_left  = output_folder / video_path.stem / "left"
        out_right = output_folder / video_path.stem / "right"

        extracted, skipped, checked = extract_frames_from_video(
            video_path, out_left, out_right,
            EXTRACT_FPS, MIN_BLUR_SCORE
        )

        grand_extracted += extracted
        grand_skipped   += skipped
        grand_checked   += checked

        print(f"  Frames checked  : {checked:,}")
        print(f"  Pairs saved     : {extracted:,}  "
              f"({extracted} left + {extracted} right = {extracted*2:,} images)")
        print(f"  Skipped (blur)  : {skipped:,}")

    total_images = grand_extracted * 2  # left + right

    print("\n" + "=" * 65)
    print("EXTRACTION COMPLETE")
    print("=" * 65)
    print(f"  Frame pairs saved     : {grand_extracted:,}")
    print(f"  Total images saved    : {total_images:,}  (left + right)")
    print(f"  Skipped (blurry)      : {grand_skipped:,}")
    print(f"  Output structure:")
    print(f"    extracted_frames/")
    print(f"    └── VBOX0001/")
    print(f"        ├── left/   ← left camera frames")
    print(f"        └── right/  ← right camera frames")
    print(f"    └── VBOX0002/")
    print(f"        ├── left/")
    print(f"        └── right/")
    print(f"    └── ...")
    print("=" * 65)
    print(f"\n  NEXT STEP:")
    print(f"  Upload frames to Roboflow for annotation.")
    print(f"  You only need to annotate E-RICKSHAWS —")
    print(f"  all other classes are already in COCO weights.")
    print("=" * 65)


if __name__ == "__main__":
    main()