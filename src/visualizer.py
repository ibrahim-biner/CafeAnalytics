import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class CafeVisualizer:
    
    # Class responsible for rendering analysis results on the screen

   
    def __init__(self,width, height):

        self.width = width
        self.height = height
        
        # Colors (B, G, R)
        self.COLOR_FREE = (0, 255, 0)     # Green (Free Table)
        self.COLOR_OCCUPIED = (0, 0, 255) # Red (Occupied Table)
        self.COLOR_PERSON = (255, 255, 0) # Cyan (Person Box)
        self.COLOR_POINT = (0, 255, 255)  # Yellow (Tracking Point)
        self.heatmap_accum = np.zeros((height, width), dtype=np.float32)

    def update_heatmap(self, x, y):
        """Adds 'heat' to the location of each person in every frame"""
        if 0 <= x < self.width and 0 <= y < self.height:
            # Add localized heat using a Gaussian-like distribution
            mask = np.zeros((self.height, self.width), dtype=np.float32)
            cv2.circle(mask, (x, y), 30, (1), -1)
            self.heatmap_accum = cv2.add(self.heatmap_accum, mask)
        
    def draw(self, frame, tracks, analyzer):
        """
        Renders visual annotations on the input frame and returns it.
        """
        out_frame = frame.copy()

        
        occupied_tables = set(analyzer.current_locations.values())
        total_tables = len(analyzer.rois) # Total table count

        for name, coords in analyzer.rois.items():
            is_occupied = name in occupied_tables
            
            if is_occupied:
                color = self.COLOR_OCCUPIED
                status_text = "Occupied"
            else:
                color = self.COLOR_FREE
                status_text = "Free"

            # We calculate the center of the table ROI for text placement
            center_x = int(np.mean([p[0] for p in coords]))
            center_y = int(np.mean([p[1] for p in coords]))

            # Text to be displayed on screen
            label = f"{name}: {status_text}"

            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            text_x = center_x - (text_w // 2)
            text_y = center_y + (text_h // 2)

            
            cv2.putText(out_frame, label, (text_x+2, text_y+2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            cv2.putText(out_frame, label, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        #  Draw people and their states 

        for track in tracks:
            track_id = track['id']
            bbox = track['bbox']
            point = track['foot_point'] 
            
            x1, y1, x2, y2 = bbox
            px, py = point

            
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), self.COLOR_PERSON, 2)

            
            # Determine current state: at table or walking
            if track_id in analyzer.current_locations:
                
                loc_name = analyzer.current_locations[track_id]
                status_label = f"{loc_name}" 
            else:
                
                status_label = "Walking"

            # Create label combining ID and state
            label = f"ID:{track_id} | {status_label}"
            
           
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out_frame, (x1, y1 - 20), (x1 + w, y1), self.COLOR_PERSON, -1)
            
            
            cv2.putText(out_frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Yellow circle at the stomach point
            cv2.circle(out_frame, (px, py), 5, self.COLOR_POINT, -1)
            cv2.circle(out_frame, (px, py), 6, (0,0,0), 1) 

            self.update_heatmap(px, py)

        # Information panel 
       
        info_text = f"Customer: {len(tracks)} | Occupied Tables: {len(occupied_tables)}/{total_tables}"
        
        cv2.rectangle(out_frame, (0,0), (550, 50), (0,0,0), -1)
        cv2.putText(out_frame, info_text, (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return out_frame
    
    def save_results(self, analyzer, output_folder="outputs"):
        """
       Generates the dashboard report:
        1. Heatmap
        2. Dashboard

        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        print("Creating a dashboard...")

        # HEATMAP 
        try:
            heatmap_norm = cv2.normalize(self.heatmap_accum, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_norm = heatmap_norm.astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(output_folder, "1_heatmap.png"), heatmap_color)
            print("Heatmap saved.")
        except: pass

        #  DASHBOARD DATA 
        # 1. Table Data
        tables = list(analyzer.table_stats.keys())
        durations = [d / 60.0 for d in analyzer.table_stats.values()] 

        # 2. Customer Statistics
        total_customers = len(analyzer.customers)
        stay_times = []
        for cust_id, data in analyzer.customers.items():
            
            stay_min = (data['last_seen'] - data['first_seen']) / 60.0
            if stay_min > 0.1: # 
                stay_times.append(stay_min)
        
        avg_stay = np.mean(stay_times) if stay_times else 0
        busiest_table = max(analyzer.table_stats, key=analyzer.table_stats.get) if durations else "Yok"

        # DASHBOARD DRAWN (2x2 Grid)
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Cafe Customer Analysis Report', fontsize=16, fontweight='bold')

        # Table Usage Times (Bar Chart)
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(tables)))
        bars = axs[0, 0].bar(tables, durations, color=colors)
        axs[0, 0].set_title("Table Usage Times (Minutes)")
        axs[0, 0].set_ylabel("Minutes")
        axs[0, 0].grid(axis='y', alpha=0.3)
        for bar in bars:
            axs[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.1f}', ha='center', va='bottom')

        # Pie Chart
        
        if sum(durations) > 0:
            axs[0, 1].pie(durations, labels=tables, autopct='%1.1f%%', startangle=90, colors=colors)
            axs[0, 1].set_title("Table Preference Distribution")
        else:
            axs[0, 1].text(0.5, 0.5, "Insufficient Data", ha='center')

        # 3.Customer Stay Times (Histogram)
        if stay_times:
            axs[1, 0].hist(stay_times, bins=10, color='skyblue', edgecolor='black')
            axs[1, 0].set_title("Customer Stay Time Distribution")
            axs[1, 0].set_xlabel("Time (Minutes)")
            axs[1, 0].set_ylabel("Number of Customers")
        else:
            axs[1, 0].text(0.5, 0.5, "No Data", ha='center')

        # 4. Text Info
        axs[1, 1].axis('off') # 
        info_text = (
            f"GENERAL STATISTICS\n"
            f"---------------------\n\n"
            f"👥 Total Customers: {total_customers}\n\n"
            f"⏱️ Average Stay: {avg_stay:.1f} min\n\n"
            f"🔥 Busiest Table: {busiest_table}\n\n"
            f"📹 Analysis Status: Completed"
        )
        axs[1, 1].text(0.1, 0.5, info_text, fontsize=14, family='monospace', va='center')

        # Kaydet
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        dash_path = os.path.join(output_folder, "2_dashboard_analiz.png")
        plt.savefig(dash_path, dpi=300) 
        plt.close()

        print(f"   ✅ Dashboard saved: {dash_path}")