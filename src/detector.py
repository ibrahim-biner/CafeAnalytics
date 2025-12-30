import cv2
from ultralytics import YOLO

class CafeTracker:
    
    # ----- Handles customer detection and tracking using YOLOv11 and ByteTrack --------

    
    def __init__(self, model_path="yolo11n.pt"):
        # We use the YOLOv11 nano model for person detection

       # We select YOLOv11n based on the available hardware, while YOLOv11s can be used for higher detection accuracy
        self.model = YOLO(model_path)
        self.person_class = 0  # In the COCO dataset, index 0 corresponds to the 'person' class


    def process_frame(self, frame):
        
        # Processes the input frame, detects persons, assigns unique IDs, and returns a list of dictionaries with tracked object data
      
        results = self.model.track(
            source=frame, 
            persist=True,   # persist=True maintains consistent object IDs across video frames
            classes=[self.person_class], 
            verbose=False
        )[0]
        
        tracked_data = []

        # If at least one ID is assigned within the bounding area, we start the processing
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            confidences = results.boxes.conf.cpu().numpy()

            for box, track_id, conf in zip(boxes, track_ids, confidences):
                x1, y1, x2, y2 = box
                
                # Computes the foot position using the bottom-center of the bounding box
                # We use the bottom-center point of the person to reduce perspective distortion
                foot_x = int((x1 + x2) / 2)
                foot_y = y2  

                tracked_data.append({
                    "id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "foot_point": (foot_x, foot_y),
                    "confidence": conf
                })
        
        return tracked_data

if __name__ == "__main__":
    print("YOLO11 Starting")
    tracker = CafeTracker()
    print("System Ready")