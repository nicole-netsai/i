import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

class ParkingDetector:
    def __init__(self, model_path='yolov8s.pt', classes_file='coco.txt'):
        self.model = YOLO(model_path)
        with open(classes_file, 'r') as f:
            self.class_list = f.read().split("\n")
        self.parking_areas = []  # To be configured by admin

    def set_parking_areas(self, areas):
        """Set the parking area polygons"""
        self.parking_areas = areas

    def process_frame(self, frame):
        """Process a single frame and return occupancy data"""
        frame = cv2.resize(frame, (1020, 500))
        results = self.model.predict(frame)
        px = pd.DataFrame(results[0].boxes.data).astype("float")
        
        area_counts = [0] * len(self.parking_areas)
        annotated_frame = frame.copy()
        
        for _, row in px.iterrows():
            x1, y1, x2, y2, _, d = map(int, row[:6])
            obj_class = self.class_list[d]
            
            if 'car' in obj_class or 'truck' in obj_class or 'bus' in obj_class:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                for i, area in enumerate(self.parking_areas):
                    if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        area_counts[i] += 1
        
        # Draw parking areas
        for i, area in enumerate(self.parking_areas):
            color = (0, 0, 255) if area_counts[i] > 0 else (0, 255, 0)
            cv2.polylines(annotated_frame, [np.array(area, np.int32)], True, color, 2)
            cv2.putText(annotated_frame, str(i+1), tuple(area[2]), 
                       cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame, area_counts
