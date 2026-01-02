import cv2
import json
import os

class CafeROISelector:
    def __init__(self, video_path, output_config="src/config.json", target_width=1280):
        self.video_path = video_path
        self.output_config = output_config
        self.target_width = target_width 
        self.points = []
        self.rois = {}
        self.scale = 1.0 

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            orig_x = int(x / self.scale)
            orig_y = int(y / self.scale)
            self.points.append((orig_x, orig_y))
            print(f"Point added (original coordinates): ({orig_x}, {orig_y})")

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Error: Video file could not be read")
            return

        orig_h, orig_w = frame.shape[:2]
        self.scale = self.target_width / float(orig_w)
        target_height = int(orig_h * self.scale)
        
        display_frame_base = cv2.resize(frame, (self.target_width, target_height))

        cv2.namedWindow("Table marker")
        cv2.setMouseCallback("Table marker", self._mouse_callback)

        print(f"\n--- Image scaled down to %{self.scale*100:.1f} ---")
        print("- Click on the table corners.")
        print("- Press 'N' to finish and name the table.")
        print("- Press 'S' to save, 'C' to clear.")

        while True:
            temp_img = display_frame_base.copy()
            
            for i, p in enumerate(self.points):
                disp_p = (int(p[0] * self.scale), int(p[1] * self.scale))
                cv2.circle(temp_img, disp_p, 4, (0, 0, 255), -1)
                if i > 0:
                    prev_p = (int(self.points[i-1][0] * self.scale), int(self.points[i-1][1] * self.scale))
                    cv2.line(temp_img, prev_p, disp_p, (0, 255, 0), 2)

            for name, pts in self.rois.items():
                scaled_pts = [(int(pt[0] * self.scale), int(pt[1] * self.scale)) for pt in pts]
                for i in range(len(scaled_pts)):
                    cv2.line(temp_img, scaled_pts[i], scaled_pts[(i+1)%len(scaled_pts)], (255, 0, 0), 2)
                cv2.putText(temp_img, name, scaled_pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Table marker", temp_img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("n"):
                if len(self.points) >= 3:
                    name = input(f"Table name (Current: {len(self.rois)}): ")
                    self.rois[name] = self.points
                    self.points = []
                else: print("Need at least 3 points to define a table ROI.")

            elif key == ord("s"):
                with open(self.output_config, "w") as f:
                    json.dump(self.rois, f, indent=4)
                print("Saved!")
                break
            
            elif key == ord("c"): self.points = []
            elif key == 27: break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    selector = CafeROISelector("data/cafe_full_analysis.mp4", target_width=1024)
    selector.run()