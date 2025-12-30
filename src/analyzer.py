import cv2
import json
import numpy as np

class CafeAnalyzer:
    
    # Analyzes customer movements, computes table occupancy durations,and manages logging

    
    def __init__(self, config_path="src/config.json", fps=30):
        self.fps = fps
        # A minimum duration of 3 seconds is required within the table area to avoid false seating detections
        self.min_duration_threshold = 3.0  
        
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

    def check_roi(self, point):

        """we determines which table a given point belongs to"""

        for name, coords in self.rois.items():
            
            pts = np.array(coords, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Point-in-polygon test: distance > 0 means the point lies inside the polygon
            if cv2.pointPolygonTest(pts, point, False) >= 0:
                return name
        return None

    def update(self, detections, frame_time):
       #Processes incoming data on each frame and updates time-based metrics
        active_ids = []

        for det in detections:
            cust_id = det['id']
            foot_point = det['foot_point']
            active_ids.append(cust_id)

            # 1. we register the customer in the system (if seen for the first time)

            if cust_id not in self.customers:
                self.customers[cust_id] = {
                    'first_seen': frame_time,
                    'last_seen': frame_time,
                    'table_sessions': [] 
                }
            else:
                self.customers[cust_id]['last_seen'] = frame_time

            # 2. Identify the table where the customer is located
            current_table = self.check_roi(foot_point)
            
            # Proceed only if the customer is at a table
            if current_table:
                
                last_known_table = self.current_locations.get(cust_id)

                if last_known_table != current_table:
                    # If the customer moves to a new table, we close the previous one and open the new session

                    self.current_locations[cust_id] = current_table
                    self.customers[cust_id]['table_sessions'].append({
                        'table': current_table,
                        'start': frame_time,
                        'end': frame_time
                    })
                else:
                    # If the customer remains at the same table, we extend the duration
                    # Update the end time of the current session

                    if self.customers[cust_id]['table_sessions']:
                        self.customers[cust_id]['table_sessions'][-1]['end'] = frame_time
            
            else:
                # The customer is currently not at any table (may be walking in the corridor)
                # If previously assigned to a table, we remove them from current_locations

                if cust_id in self.current_locations:
                    del self.current_locations[cust_id]

    def generate_report(self, output_path="outputs/customer_log.txt", total_video_duration=0):
        """We generate the log report"""

        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("CAFE CUSTOMER ANALYTICS LOG\n")
            f.write("===========================\n")
            
            
            total_customers = len(self.customers)
            f.write(f"Processed Video Duration: {total_video_duration:.2f}s\n\n")
            f.write("CUSTOMER RECORDS\n")
            
            for cust_id, data in self.customers.items():
                f.write(f"Customer ID: {cust_id}\n")
                
                
                fs = self._format_time(data['first_seen'])
                ls = self._format_time(data['last_seen'])
                total_dur = data['last_seen'] - data['first_seen']
                
                f.write(f"First Seen: {fs}\n")
                f.write(f"Last Seen: {ls}\n")
                f.write(f"Total Appearance Time: {self._format_time(total_dur, is_duration=True)}\n")
                
                f.write("Table Usage:\n")
                has_valid_usage = False
                
                
                usage_summary = {} # { 'Table-1': total_seconds }
                
                for session in data['table_sessions']:
                    duration = session['end'] - session['start']
                    if duration >= self.min_duration_threshold:
                        tbl = session['table']
                        usage_summary[tbl] = usage_summary.get(tbl, 0) + duration
                        
                        
                        # f.write(f"  * {tbl}: {self._format_time(duration, True)} ({self._format_time(session['start'])} - {self._format_time(session['end'])})\n")

                for tbl, dur in usage_summary.items():
                    f.write(f"  * {tbl}: {self._format_time(dur, True)}\n")
                    
                    self.table_stats[tbl] += dur
                    has_valid_usage = True
                
                if not has_valid_usage:
                    f.write("  * (No significant table usage detected)\n")
                
                # Status 
                status = "Still inside" if abs(data['last_seen'] - total_video_duration) < 2.0 else "Left"
                f.write(f"Status: {status}\n")
                f.write("-" * 30 + "\n")

            # TABLE USAGE STATISTICS
            f.write("\nTABLE USAGE STATISTICS\n")
            has_valid_usage = False
                
               
                
            for session in data['table_sessions']:
                duration = session['end'] - session['start']
                
                if duration >= self.min_duration_threshold:
                    tbl = session['table']
                    
                    start_fmt = self._format_time(session['start'])
                    end_fmt = self._format_time(session['end'])
                    dur_fmt = self._format_time(duration, True)
                    
                    f.write(f"  * {tbl}: {dur_fmt} ({start_fmt} - {end_fmt})\n")
                    
                    self.table_stats[tbl] += duration
                    has_valid_usage = True
            
            if not has_valid_usage:
                f.write("  * (No significant table usage detected)\n")
    
    def _format_time(self, seconds, is_duration=False):
        
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if is_duration:
            return f"{int(m)}m {int(s)}s"
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"