import random
import cv2
from typing import List, Tuple
from detective import Detection

class BoxGeometry:
    """Helper class for bounding box calculations."""

    @staticmethod
    def compute_iou(box_a: Tuple[int, int, int, int], 
                    box_b: Tuple[int, int, int, int]) -> float:
        """
        Compute Intersection over Union between two boxes.
        
        Args:
            box_a: (x1, y1, x2, y2)
            box_b: (x1, y1, x2, y2)
        
        Returns:
            IoU value between 0 and 1
        """
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])

        inter_w = max(0, xb - xa)
        inter_h = max(0, yb - ya)
        inter = inter_w * inter_h

        if inter == 0:
            return 0.0

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    @staticmethod
    def compute_ioa(box_a: Tuple[int, int, int, int], 
                    box_b: Tuple[int, int, int, int]) -> float:
        """
        Compute Intersection over Area (smaller area) between two boxes.
        
        Args:
            box_a: (x1, y1, x2, y2)
            box_b: (x1, y1, x2, y2)
        
        Returns:
            IoA value between 0 and 1
        """
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])

        inter_w = max(0, xb - xa)
        inter_h = max(0, yb - ya)
        inter = inter_w * inter_h

        if inter == 0:
            return 0.0

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        area_min = min(area_a, area_b)

        return inter / area_min if area_min > 0 else 0.0

    @staticmethod
    def compute_centroid(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        Compute centroid of a bounding box.
        
        Args:
            box: (x1, y1, x2, y2)
        
        Returns:
            (center_x, center_y)
        """
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        return (cx, cy)

    @staticmethod
    def compute_centroid_distance(box_a: Tuple[int, int, int, int],
                                  box_b: Tuple[int, int, int, int]) -> float:
        """
        Compute Euclidean distance between centroids of two boxes.
        
        Args:
            box_a: (x1, y1, x2, y2)
            box_b: (x1, y1, x2, y2)
        
        Returns:
            Distance in pixels
        """
        cx_a, cy_a = BoxGeometry.compute_centroid(box_a)
        cx_b, cy_b = BoxGeometry.compute_centroid(box_b)
        distance = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5
        return distance

    @staticmethod
    def normalize_box(box: Tuple[int, int, int, int],
                     frame_width: int,
                     frame_height: int) -> Tuple[float, float, float, float]:
        """
        Normalize box coordinates to [0, 1] range.
        Returns (center_x_norm, center_y_norm, width_norm, height_norm).
        
        Args:
            box: (x1, y1, x2, y2)
            frame_width: Width of frame in pixels
            frame_height: Height of frame in pixels
        
        Returns:
            (cx_norm, cy_norm, width_norm, height_norm)
        """
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2 / frame_width
        cy = (y1 + y2) / 2 / frame_height
        bw = (x2 - x1) / frame_width
        bh = (y2 - y1) / frame_height
        return (cx, cy, bw, bh)


class BoxDrawer:
    """Helper class for drawing bounding boxes on frames."""

    # Default color palette (BGR format)
    DEFAULT_COLORS = {
        'erickshaw': (0, 255, 0),      # green
        'person': (255, 178, 50),       # blue
        'bicycle': (0, 165, 255),       # orange
        'car': (0, 0, 255),             # red
        'motorcycle': (255, 0, 255),    # magenta
        'bus': (255, 255, 0),           # cyan
        'truck': (128, 0, 128),         # purple
    }

    @staticmethod
    def draw_box(frame: cv2.Mat,
                 x1: int, y1: int, x2: int, y2: int,
                 label: str,
                 confidence: float,
                 color: Tuple[int, int, int] = None) -> cv2.Mat:
        """
        Draw a bounding box with label on a frame.
        
        Args:
            frame: Input frame (modified in-place)
            x1, y1, x2, y2: Box coordinates
            label: Text label (e.g., "person#1")
            confidence: Confidence score to display
            color: (B, G, R) tuple. If None, uses class-based color.
        
        Returns:
            Modified frame
        """
        if color is None:
            color = BoxDrawer.DEFAULT_COLORS.get(label, (200, 200, 200))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {confidence:.2f}"
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

    @staticmethod
    def draw_detections(frame: cv2.Mat,
                       detections: List[Detection],
                       use_random_colors: bool = False) -> cv2.Mat:
        """
        Draw multiple detections on a frame.
        
        Args:
            frame: Input frame
            detections: List of Detection objects
            use_random_colors: If True, use random color per tracking_id
        
        Returns:
            Annotated frame
        """
        for det in detections:
            if use_random_colors and det.tracking_id is not None:
                random.seed(det.tracking_id)
                color = (random.randint(0, 255), random.randint(0, 255),
                        random.randint(0, 255))
            else:
                color = BoxDrawer.DEFAULT_COLORS.get(det.label, (200, 200, 200))

            label = det.object_id if det.object_id else det.label
            BoxDrawer.draw_box(frame, det.x1, det.y1, det.x2, det.y2,
                             label, det.confidence, color)

        return frame

