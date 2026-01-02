import cv2
import json
import numpy as np

class CafeAnalyzer:
    
    # Analyzes customer movements, computes table occupancy durations,and manages logging

    
    def __init__(self, config_path="src/config.json", fps=30):
        self.fps = fps
        # A minimum duration of 3 seconds is required within the table area to avoid false seating detections
        self.min_duration_threshold = 4.0  

        # --- ID STABILIZATION MEMORY ---
        self.recent_tracks = {}   # ACTIVE + GHOST ID'ler
        self.GHOST_TTL = 5.0      # saniye
        self.MERGE_TIME_TH = 1.0  # saniye
        self.MERGE_DIST_TH = 60   # pixel (720p için ideal)

        
        # Loads the predefined table positions (ROIs)

        with open(config_path, "r") as f:
            self.rois = json.load(f)
        
        """
            In-memory database. we stored data in RAM

            Structure:
            {
            customer_id: {
                'first_seen': 0,
                'last_seen': 0,
                'tables': {
                    'Table-1': [duration, start_time, end_time]
                }
            }
            }
    """
        self.customers = {} 
        
        # Current state mapping
        self.current_locations = {} 
        
        # Maintains statistics for each table
        self.table_stats = {name: 0.0 for name in self.rois.keys()}

        self.patience_counters = {}
        self.PATIENCE_LIMIT = int(fps * 5.0)

    def check_roi(self, point):

        """we determines which table a given point belongs to"""

        for name, coords in self.rois.items():
            
            pts = np.array(coords, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Point-in-polygon test: distance > 0 means the point lies inside the polygon
            if cv2.pointPolygonTest(pts, point, False) >= 0:
                return name
        return None


    def _match_previous_id(self, point, table, frame_time, mean_color=None):
        """
        Matches a new track to a previous ID (ACTIVE or GHOST)
        using time, distance, table, velocity, and appearance cues

        """
        for old_id, data in self.recent_tracks.items():
            time_diff = frame_time - data['last_seen']
            ox, oy = data['last_point']
            dist = np.hypot(point[0] - ox, point[1] - oy)

            #  Velocity consistency check to avoid ID switches due to sudden direction changes 
            old_velocity = data.get('velocity', (0,0))  
            new_velocity = (point[0] - ox, point[1] - oy)  
            dot = old_velocity[0]*new_velocity[0] + old_velocity[1]*new_velocity[1]  
            mag_old = (old_velocity[0]**2 + old_velocity[1]**2)**0.5 
            mag_new = (new_velocity[0]**2 + new_velocity[1]**2)**0.5  
            if mag_old > 0 and mag_new > 0:
                cos_angle = dot / (mag_old*mag_new)
                if cos_angle < 0.5:  # we ignore this match if motion direction changes abruptly
                    continue  

            # Basic appearance similarity check using color features
            if mean_color is not None and 'mean_color' in data:
                color_diff = np.linalg.norm(data['mean_color'] - mean_color)
                if color_diff > 50:  
                    continue  

            # Distance, time, and table consistency check 
            if (dist < self.MERGE_DIST_TH or time_diff < self.MERGE_TIME_TH) and data['table'] == table:
                return old_id

        return None




    def update(self, detections, frame_time,frame):
        """
        Updates customer states for the current video frame
        Handles customer tracking, table assignment, and cleanup logic

        """
        
        # IDs detected in the current frame 
        present_ids = set()

        # PROCESS VISIBLE CUSTOMERS
        for det in detections:
            
            incoming_id = det['id']
            foot_point = det['foot_point']
            current_table = self.check_roi(foot_point)

            # Compute basic appearance features for ID stabilization
            x1, y1, x2, y2 = det['bbox']  
            roi = frame[y1:y2, x1:x2]     
            mean_color = roi.mean(axis=(0,1))  
            det['mean_color'] = mean_color     

            # ID MERGE CHECK 
            matched_id = self._match_previous_id(
                foot_point,
                current_table,
                frame_time,
                mean_color=mean_color
            )

            cust_id = matched_id if matched_id is not None else incoming_id

            
            if cust_id in self.recent_tracks:  
                last_pt = self.recent_tracks[cust_id]['last_point']  
                velocity = (foot_point[0] - last_pt[0], foot_point[1] - last_pt[1])  
            else:
                velocity = (0,0)  


            foot_point = det['foot_point']
            present_ids.add(cust_id) # We mark this customer as visible in the current frame

            # We register customer if seen for the first time
            if cust_id not in self.customers:
                self.customers[cust_id] = {
                    'first_seen': frame_time, 'last_seen': frame_time, 'table_sessions': []
                }
            else:
                self.customers[cust_id]['last_seen'] = frame_time

            # We determine which table the customer is currently in
            current_table = self.check_roi(foot_point)
            
            if current_table:
                # Customer is detected inside a table ROI, we reset patience counter
                self.patience_counters[cust_id] = self.PATIENCE_LIMIT
                
                last_known = self.current_locations.get(cust_id)
                if last_known != current_table:
                    self.current_locations[cust_id] = current_table
                    self.customers[cust_id]['table_sessions'].append({
                        'table': current_table, 'start': frame_time, 'end': frame_time
                    })
                else:
                    if self.customers[cust_id]['table_sessions']:
                        self.customers[cust_id]['table_sessions'][-1]['end'] = frame_time

            self.recent_tracks[cust_id] = {
                'last_point': foot_point,
                'last_seen': frame_time,
                'table': current_table,
                'velocity': velocity,   
                'mean_color': mean_color
                }



        

        # HANDLE INVISIBLE OR TEMPORARILY LOST CUSTOMERS
        ids_to_remove = []

        for cust_id in list(self.current_locations.keys()):
            # Customer not detected in the current frame
            if cust_id not in present_ids:

                if cust_id in self.recent_tracks:
                 self.recent_tracks[cust_id]['last_seen'] = frame_time
                
                 # We apply patience mechanism to avoid flickering detections
                if self.patience_counters.get(cust_id, 0) > 0:
                    self.patience_counters[cust_id] -= 1
                    
                  
                    if self.customers[cust_id]['table_sessions']:
                        self.customers[cust_id]['table_sessions'][-1]['end'] = frame_time
                
                else:
                    
                    ids_to_remove.append(cust_id)

        # We remove customers after iteration to avoid runtime errors
        for cust_id in ids_to_remove:
            del self.current_locations[cust_id]

        to_delete = []
        for old_id, data in self.recent_tracks.items():
            if frame_time - data['last_seen'] > self.GHOST_TTL:
                to_delete.append(old_id)

        for old_id in to_delete:
            del self.recent_tracks[old_id]

    def generate_report(self, output_path="outputs/customer_log.txt", total_video_duration=0):
        """
        Generates the log report.

        Includes:
        - Customer-level details (first/last seen, table usage intervals)
        - Table usage statistics (including usage percentages)
        - Overall summary metrics

        """

        # 1. COMPUTE GLOBAL STATISTICS
        
        self.table_stats = {name: 0.0 for name in self.rois.keys()}
        
        active_customers = 0
        total_stay_duration = 0
        total_customers = len(self.customers)

        # Preprocess customer data for table statistics and summary metrics
        for cust_id, data in self.customers.items():
             # We determine if the customer is still inside the scene
            if abs(data['last_seen'] - total_video_duration) < 2.0:
                active_customers += 1
            
            # We accumulate total appearance duration
            total_stay_duration += (data['last_seen'] - data['first_seen'])

            #We accumulate valid table usage durations
            for session in data['table_sessions']:
                duration = session['end'] - session['start']
                if duration >= self.min_duration_threshold:
                    self.table_stats[session['table']] += duration

        # 2. WRITE REPORT TO FILE
        with open(output_path, "w", encoding="utf-8") as f:
            
            # --- CUSTOMER RECORDS ---
            f.write("CUSTOMER RECORDS\n")
            
            for cust_id, data in self.customers.items():
                # Format customer ID as zero-padded (e.g., 001, 002)
                try:
                    fmt_id = f"{int(cust_id):03d}"
                except:
                    fmt_id = str(cust_id)
                
                f.write(f"Customer ID: {fmt_id}\n")
                
                fs = self._format_time(data['first_seen'])
                ls = self._format_time(data['last_seen'])
                total_dur = data['last_seen'] - data['first_seen']
                
                f.write(f"  - First Seen: {fs}\n")
                f.write(f"  - Last Seen: {ls}\n")
                f.write(f"  - Total Appearance Time: {self._format_time(total_dur, is_duration=True)}\n")
                
                f.write("  - Table Usage:\n")
                has_valid_usage = False
                
                 # We write each valid table session with time interval details
                for session in data['table_sessions']:
                    duration = session['end'] - session['start']
                    if duration >= self.min_duration_threshold:
                        tbl = session['table']
                        dur_fmt = self._format_time(duration, True)
                        start_fmt = self._format_time(session['start'])
                        end_fmt = self._format_time(session['end'])
                        
                         # Format like : * Table-3: 7m 45s (00:02:15 - 00:10:00)
                        f.write(f"    * {tbl}: {dur_fmt} ({start_fmt} - {end_fmt})\n")
                        has_valid_usage = True
                
                if not has_valid_usage:
                    f.write("    * (No significant table usage)\n")
                
                #We determine customer status
                status = "Still inside" if abs(data['last_seen'] - total_video_duration) < 2.0 else "Left"
                f.write(f"  - Status: {status}\n")
                f.write("\n") # Müşteriler arası boşluk

            # TABLE USAGE STATISTICS 
            f.write("--- TABLE USAGE STATISTICS --\n")
            
            
            busiest_table_name = "None"
            max_usage = -1

            #  Report statistics for all defined tables
            for tbl_name in self.rois.keys():
                usage_seconds = self.table_stats.get(tbl_name, 0.0)
                
                # We compute usage percentage relative to video duration
                if total_video_duration > 0:
                    usage_percent = int((usage_seconds / total_video_duration) * 100)
                else:
                    usage_percent = 0
                
                # We track the busiest table
                if usage_seconds > max_usage:
                    max_usage = usage_seconds
                    busiest_table_name = tbl_name
                
                # Format: Table-1: Total 0m 00s (Usage: %0)
                dur_fmt = self._format_time(usage_seconds, True)
                f.write(f"{tbl_name}: Total {dur_fmt} (Usage: %{usage_percent})\n")

            # SUMMARY)
            f.write("\n--- SUMMARY --\n")
            f.write(f"Total Detected Customers: {total_customers}\n")
            f.write(f"Active Customers: {active_customers}\n")
            
            # Average stay duration
            if total_customers > 0:
                avg_stay_sec = total_stay_duration / total_customers
                f.write(f"Average Stay Duration: {self._format_time(avg_stay_sec, True)}\n")
            else:
                f.write("Average Stay Duration: 0m 0s\n")
            
            
            if max_usage <= 0:
                busiest_table_name = "None"
                
            f.write(f"Busiest Table: {busiest_table_name}\n")
    
    def _format_time(self, seconds, is_duration=False):
        
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if is_duration:
            return f"{int(m)}m {int(s)}s"
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"