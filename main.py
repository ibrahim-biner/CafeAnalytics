import cv2
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.detector import CafeTracker
from src.analyzer import CafeAnalyzer
from src.visualizer import CafeVisualizer

def main():
    # --- SETTİNGS ---
    VIDEO_PATH = "data/cafe_full_analysis.mp4"  # The video we combined
    CONFIG_PATH = "src/config.json"             # The tables we drew
    OUTPUT_LOG = "outputs/customer_log.txt"     
    OUTPUT_VIDEO = "outputs/processed_demo.mp4" 

    
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found -> {VIDEO_PATH}")
        return
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Config not found -> {CONFIG_PATH}")
        return

    # --- system startup ---
    print("System starting up...")
    
    # 1.Detector (YOLO11 + ByteTrack)
    tracker = CafeTracker(model_path="yolo11m-pose.pt")
    
    # 2. video settings
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 3. Analysis engine
    # FPS is read dynamically from the video to ensure accurate time calculations

    analyzer = CafeAnalyzer(config_path=CONFIG_PATH, fps=fps)


    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    
    visualizer = CafeVisualizer(video_width, video_height)

   # 4. Video recorder
    writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print(f"Analysis started: {total_frames} frames will be processed")
    print("Press 'q' to exit.\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break 
        
        frame_count += 1
        current_time_sec = frame_count / fps 

        # A. DETECTION AND TRACKING (detection layer)
        # Returns a list of {id, bbox, foot_point}
        tracks = tracker.process_frame(frame)

        # B. Analysis layer
        # Computes person-to-table assignments
        analyzer.update(tracks, current_time_sec,frame)

        output_frame = visualizer.draw(frame, tracks, analyzer)

    

        # D. OUTPUT RECORDING AND VISUALIZATION
        writer.write(output_frame)
        
        display_frame = cv2.resize(output_frame, (1280, 720)) 
        cv2.imshow("Cafe Analytics Live", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User interrupted.")
            break

    # --- CLEANUP AND EXIT ---

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # Create final report
    video_duration = frame_count / fps
    analyzer.generate_report(OUTPUT_LOG, total_video_duration=video_duration)


    print("Graphic and Heatmap saving...")
    # Save statistics and heatmap outputs to disk
    visualizer.save_results(analyzer, output_folder="outputs")
    
    
    
    print("\n--- ANALYSIS FINISHED ---")
    print(f"Raport saved: {OUTPUT_LOG}")
    print(f"Processed video: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()