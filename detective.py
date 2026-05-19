import cv2
from typing import Dict, Optional, List
from ultralytics import YOLO
from dataclasses import dataclass


@dataclass
class Detection:
    """Represents a single object detection."""
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    tracking_id: Optional[int] = None
    object_id: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert detection to dictionary."""
        return {
            'label': self.label,
            'confidence': self.confidence,
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2,
            'tracking_id': self.tracking_id,
            'object_id': self.object_id,
        }


class YOLODetector:
    """Wrapper for YOLO model management."""

    def __init__(self, model_path: str):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to .pt model file
        """
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def detect(self, frame: cv2.Mat, conf_threshold: float = 0.4, object_classes: Optional[List] = None) -> List[Detection]:
        """
        Run detection on frame.
        
        Args:
            frame: Input frame
            conf_threshold: Confidence threshold
            object_classes: Class ID of object under detective

        
        Returns:
            List of Detection objects
        """
        results = self.model(frame, conf=conf_threshold, verbose=False)
        detections = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])

                if object_classes is not None and cls_id not in object_classes:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(Detection(
                    label=self.class_names[cls_id],
                    confidence=float(box.conf[0]),
                    x1=x1, y1=y1, x2=x2, y2=y2
                ))

        return detections

    def track(self, frame: cv2.Mat, conf_threshold: float = 0.4, object_classes: Optional[List] = None) -> List[Detection]:
        """
        Run tracking on frame.
        
        Args:
            frame: Input frame
            conf_threshold: Confidence threshold
            object_classes: Class ID of object under tracking
        
        Returns:
            List of Detection objects with tracking IDs
        """
        results = self.model.track(frame, conf=conf_threshold,
                                  verbose=False, persist=True, tracker='custom_botsort.yaml')
        detections = []

        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes
            track_ids = result.boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                cls_id = int(box.cls[0])

                if object_classes is not None and cls_id not in object_classes:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = self.class_names[cls_id]
                detections.append(Detection(
                    label=label,
                    confidence=float(box.conf[0]),
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    tracking_id=track_id,
                    object_id=f"{label}#{track_id}"
                ))

        return detections
