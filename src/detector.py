import cv2
import numpy as np
from ultralytics import YOLO

class CafeTracker:
    def __init__(self, model_path="yolo11m-pose.pt"): 
        # Using YOLO pose estimation model
        print(f"Model loading : {model_path}...")
        self.model = YOLO(model_path)
        self.person_class = 0 

    def process_frame(self, frame):
        """
        Pose-based tracking with abdomen-level reference point computation
        
        """
        # 1. Tracking process (BoT-SORT with optimized parameters)
        results = self.model.track(
            source=frame, 
            persist=True, 
            classes=[0],    
            conf=0.22,      
            iou=0.7,        
            imgsz=960,      
            tracker="src/custom_bytetrack.yaml", 
            verbose=False
        )[0]
        
        tracked_data = []

        if results.boxes.id is not None:
            # Fetching data
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            confidences = results.boxes.conf.cpu().numpy()
            
            # Fetch keypoints
            # Shape: (N, 17, 3) -> [x, y, conf]
            keypoints = results.keypoints.data.cpu().numpy()

            for i, track_id in enumerate(track_ids):
                x1, y1, x2, y2 = boxes[i]
                kpts = keypoints[i]
                
                

               # Pose-based reference point calculation at torso level 
               # Shoulder keypoints: index 5 (left) and index 6 (right)
                left_shoulder = kpts[5]
                right_shoulder = kpts[6]
                
                # If both shoulders are confidently detected (confidence > 0.5)
                if left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5:
                    
                    # We calculate the midpoint of the left and right shoulders
                    shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
                    shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
                    
                   # We compute the bounding box height
                    box_height = y2 - y1
                    
                    # C. Apply offset: move the point downward (towards the torso)
                    # Move down from shoulder level by 25% of the person's height
                    # This helps the point align accurately with the table

                    
                    point_x = int(shoulder_center_x)
                    point_y = int(shoulder_center_y + (box_height * 0.25))
                
                else:
                    # If shoulders are not visible
                    # We use a point at 40% from the top of the bounding box
                    point_x = int((x1 + x2) / 2)
                    point_y = int(y1 + (y2 - y1) * 0.40)

                tracked_data.append({
                    "id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "foot_point": (point_x, point_y), 
                    "confidence": confidences[i]
                })
        
        return tracked_data    