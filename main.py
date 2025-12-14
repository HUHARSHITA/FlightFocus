import threading
import time
import random
import queue
import os
import math
import json
from datetime import datetime
import cv2
from PIL import Image, ImageTk, ImageDraw
import customtkinter as ctk
import pygame
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================
VIDEO_PATH = "plane.mp4"
ALARM_SOUND = "alarm.mp3"   
CHEER_SOUND = "cheer.mp3"
MUSIC_PATH = "lofi.mp3"
ROLL_SOUND = "roll.mp3"

# ==========================================
# 1. MEDIA MANAGER
# ==========================================
class MediaManager:
    @staticmethod
    def init():
        pygame.mixer.init()

    @staticmethod
    def play_music():
        if os.path.exists(MUSIC_PATH):
            try:
                # Play bg music if not already playing
                if not pygame.mixer.Channel(0).get_busy():
                    pygame.mixer.Channel(0).play(pygame.mixer.Sound(MUSIC_PATH), loops=-1)
                    pygame.mixer.Channel(0).set_volume(0.4)
            except: pass

    @staticmethod
    def stop_music():
        pygame.mixer.Channel(0).stop()

    @staticmethod
    def start_alarm():
        if os.path.exists(ALARM_SOUND):
            try:
                if not pygame.mixer.Channel(1).get_busy():
                    pygame.mixer.Channel(1).play(pygame.mixer.Sound(ALARM_SOUND), loops=-1)
            except: pass

    @staticmethod
    def stop_alarm():
        pygame.mixer.Channel(1).stop()

    @staticmethod
    def play_sfx(path):
        if os.path.exists(path):
            pygame.mixer.Channel(2).play(pygame.mixer.Sound(path))

    @staticmethod
    def play_roll():
        if os.path.exists(ROLL_SOUND):
            try:
                pygame.mixer.Channel(3).play(pygame.mixer.Sound(ROLL_SOUND), loops=-1)
            except: pass

    @staticmethod
    def stop_roll():
        pygame.mixer.Channel(3).stop()

# ==========================================
# 2. LOG MANAGER
# ==========================================
class LogManager:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    FILE_PATH = os.path.join(SCRIPT_DIR, "flight_log.json")

    @staticmethod
    def save_trip(destination, duration, status):
        entry = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "source": "DELHI",
            "destination": destination,
            "duration": round(duration, 2),
            "status": status
        }
        
        data = []
        if os.path.exists(LogManager.FILE_PATH):
            try:
                with open(LogManager.FILE_PATH, "r") as f:
                    content = f.read()
                    if content: data = json.loads(content)
            except: pass
            
        data.append(entry)
        
        try:
            with open(LogManager.FILE_PATH, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Save Error: {e}")

    @staticmethod
    def get_logs():
        if not os.path.exists(LogManager.FILE_PATH):
            return []
        try:
            with open(LogManager.FILE_PATH, "r") as f:
                return json.load(f)
        except: return []

# ==========================================
# 3. MONITOR (AI VISION - UPDATED)
# ==========================================
class Monitor(threading.Thread):
    def __init__(self, status_queue, pip_queue, webcam_index=0):
        super().__init__(daemon=True)
        self.status_queue = status_queue
        self.pip_queue = pip_queue
        self.webcam_index = webcam_index
        self.running = True
        self.crashed = False
        
        # Buffer
        self.absence_frames = 0      
        self.ABSENCE_TOLERANCE = 10  # Slightly faster reaction
        
        # State
        self.warning_active = False
        self.warning_start_time = 0
        self.last_warning_int = 0
        
        print("⏳ Loading AI Model...")
        self.model = YOLO("yolov8n.pt") 
        print("✅ AI Ready.")

    def run(self):
        cap = cv2.VideoCapture(self.webcam_index)
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            results = self.model(frame, verbose=False, conf=0.4)
            found_person = False
            found_phone = False

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    
                    if cls_name == "person":
                        found_person = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "PILOT", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                    elif cls_name in ["cell phone", "mobile phone"]:
                        found_phone = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, "PHONE", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            try:
                small_frame = cv2.resize(frame, (240, 180))
                if self.pip_queue.empty(): 
                    self.pip_queue.put(small_frame)
            except: pass

            current_time = time.time()
            
            # --- UPDATED LOGIC: UNIFIED DISTRACTION HANDLING ---
            
            # Determine if there is ANY distraction
            distraction_reason = None
            if found_phone:
                distraction_reason = "PHONE DETECTED"
            elif not found_person:
                distraction_reason = "PILOT ABSENCE"
            
            # If Distracted (Phone OR No Person)
            if distraction_reason and not self.crashed:
                self.absence_frames += 1
                
                # Buffer Check
                if self.absence_frames > self.ABSENCE_TOLERANCE:
                    if not self.warning_active:
                        # START WARNING
                        self.warning_active = True
                        self.warning_start_time = current_time
                    else:
                        # CONTINUE WARNING
                        elapsed = current_time - self.warning_start_time
                        remaining = 15 - int(elapsed)
                        
                        if remaining <= 0:
                            # Time's up -> Crash
                            self.trigger_crash(distraction_reason)
                        else:
                            # Update UI (only if second changed)
                            if remaining != self.last_warning_int:
                                self.last_warning_int = remaining
                                # Send Tuple: (Time, Reason)
                                self.status_queue.put(("WARNING", (remaining, distraction_reason)))
            
            # Else Safe (Person AND No Phone)
            else:
                self.absence_frames = 0 
                if self.warning_active and not self.crashed:
                    # Clear Warning
                    self.warning_active = False
                    self.status_queue.put(("CLEAR_WARNING", None))

            time.sleep(0.05) 
        cap.release()

    def trigger_crash(self, reason):
        self.crashed = True
        self.warning_active = False
        self.status_queue.put(("CRASH", reason))

    def stop(self):
        self.running = False

# ==========================================
# 4. VIDEO PLAYER
# ==========================================
class VideoPlayer(threading.Thread):
    def __init__(self, video_queue, path):
        super().__init__(daemon=True)
        self.video_queue = video_queue
        self.path = path
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = 1 / fps
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            if self.video_queue.qsize() < 2:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.video_queue.put(rgb)
            
            time.sleep(delay)
        cap.release()

    def stop(self):
        self.running = False

# ==========================================
# 5. GUI APPLICATION
# ==========================================
class FocusApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        MediaManager.init()
        
        self.title("Flight Focus ✈️")
        self.geometry("1100x800")
        ctk.set_appearance_mode("Dark")
        self.bind("<Escape>", self.toggle_fullscreen)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.cities = {
            "Pune": 2.0, "Mumbai": 2.5, "Udipi": 2.75, 
            "Goa": 2.5, "Ladakh": 1.0, "Kochi": 3.25,
            "Jaipur": 1.5, "Kolkata": 2.25, "Chennai": 3.0
        }
        self.city_keys = list(self.cities.keys())
        self.colors = ["#aaf77e",'#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#FFD93D', '#6C5B7B', '#F7A072']
        
        self.status_queue = queue.Queue()
        self.pip_queue = queue.Queue(maxsize=1) 
        self.bg_video_queue = queue.Queue(maxsize=2)
        
        self.monitor = None
        self.player = None

        self.container = ctk.CTkFrame(self)
        self.container.pack(fill="both", expand=True)
        
        self._init_setup_screen()
        self._init_flight_screen()
        
        self.show_setup()
        
        self.after(33, self._render_loop)
        self.is_warning_visible = False

    def _render_loop(self):
        if not self.bg_video_queue.empty():
            try:
                frame = self.bg_video_queue.get_nowait()
                w, h = self.winfo_screenwidth(), self.winfo_screenheight()
                img = Image.fromarray(frame)
                img = img.resize((w, h), Image.Resampling.NEAREST) 
                ctk_img = ctk.CTkImage(img, size=(w, h))
                self.vid_lbl.configure(image=ctk_img)
            except: pass

        if not self.pip_queue.empty():
            try:
                frame = self.pip_queue.get_nowait()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                ctk_img = ctk.CTkImage(img, size=(240, 180))
                self.pip_label.configure(image=ctk_img)
            except: pass

        try:
            while True:
                msg, data = self.status_queue.get_nowait()
                if msg == "CRASH": self.show_crash(data)
                elif msg == "WARNING": self.show_warning(data)
                elif msg == "CLEAR_WARNING": self.hide_warning()
        except queue.Empty: pass

        self.after(33, self._render_loop)

    def toggle_fullscreen(self, event=None):
        current_state = self.attributes("-fullscreen")
        self.attributes("-fullscreen", not current_state)

    def get_elapsed_hours(self):
        if not hasattr(self, 'remaining'): return 0.0
        total_seconds = int(self.flight_time * 3600)
        elapsed_seconds = total_seconds - self.remaining
        return elapsed_seconds / 3600.0

    def return_to_home(self):
        """Switches UI back to setup and FIXES AUDIO"""
        # 1. Stop Alarm
        MediaManager.stop_alarm()
        
        # 2. Ensure BG Music is Playing
        MediaManager.play_music()
        
        # 3. Switch Screens
        self.flight_frame.pack_forget()
        self.setup_frame.pack(fill="both", expand=True)
        
        # 4. Reset States
        self.btn_spin.configure(state="normal")
        self.attributes("-fullscreen", False)

    # ----------------------------------------
    # SETUP SCREEN
    # ----------------------------------------
    def _init_setup_screen(self):
        self.setup_frame = ctk.CTkFrame(self.container, fg_color="transparent")
        
        header_frame = ctk.CTkFrame(self.setup_frame, fg_color="transparent")
        header_frame.pack(pady=(20, 10))
        ctk.CTkLabel(header_frame, text="DESTINATION PICKER", font=("Impact", 40)).pack(side="left")

        self.wheel_container = ctk.CTkFrame(self.setup_frame, fg_color="transparent")
        self.wheel_container.pack(expand=True, fill="both")

        self.wheel_size = 350
        self.wheel_lbl = ctk.CTkLabel(self.wheel_container, text="")
        self.wheel_lbl.place(relx=0.5, rely=0.45, anchor="center") 
        
        self.base_wheel = self._draw_wheel_base(self.wheel_size)
        self._rotate_wheel(0)

        self.arrow_lbl = ctk.CTkLabel(self.setup_frame, text="▼", font=("Arial", 50), text_color="#FF4757")
        self.arrow_lbl.place(relx=0.5, rely=0.20, anchor="center")

        self.controls_frame = ctk.CTkFrame(self.wheel_container, fg_color="transparent")
        self.controls_frame.place(relx=0.5, rely=0.8, anchor="center")

        self.btn_spin = ctk.CTkButton(self.controls_frame, text="SPIN WHEEL", font=("Arial", 20, "bold"), 
                                      height=50, width=200, fg_color="#2ED573", hover_color="#2CC069",
                                      command=self.spin_mechanics)
        self.btn_spin.pack(side="left", padx=10)
        
        self.btn_log = ctk.CTkButton(self.controls_frame, text="📖 LOGBOOK", font=("Arial", 14, "bold"), 
                                     height=50, width=150, fg_color="#45B7D1", hover_color="#2D98B2",
                                     command=self.open_logbook_overlay)
        self.btn_log.pack(side="left", padx=10)
        
        self.popup_frame = ctk.CTkFrame(self.setup_frame, fg_color="#2b2b2b", border_width=2, border_color="white", width=400, height=300)
        self.logbook_frame = ctk.CTkFrame(self.setup_frame, fg_color="#2b2b2b", border_width=2, border_color="white", width=800, height=600)

    def show_setup(self):
        self.flight_frame.pack_forget()
        self.setup_frame.pack(fill="both", expand=True)

    def _draw_wheel_base(self, size):
        img = Image.new("RGBA", (size, size), (0,0,0,0))
        draw = ImageDraw.Draw(img)
        n = len(self.city_keys)
        arc = 360 / n
        for i, city in enumerate(self.city_keys):
            start = i * arc
            end = (i+1) * arc
            draw.pieslice([10,10,size-10,size-10], start=start, end=end, fill=self.colors[i%len(self.colors)], outline="white")
            mid_rad = math.radians(start + arc/2)
            dist = size * 0.35
            tx = size/2 + dist * math.cos(mid_rad)
            ty = size/2 + dist * math.sin(mid_rad)
            draw.text((tx-20, ty-10), city.upper(), fill="black", font_size=14)
        return img

    def _rotate_wheel(self, angle):
        rotated = self.base_wheel.rotate(angle, resample=Image.Resampling.BICUBIC)
        ctk_img = ctk.CTkImage(rotated, size=(self.wheel_size, self.wheel_size))
        self.wheel_lbl.configure(image=ctk_img)

    def spin_mechanics(self):
        self.btn_spin.configure(state="disabled")
        MediaManager.play_roll()
        self.selected_city = random.choice(self.city_keys)
        self.flight_time = self.cities[self.selected_city]
        
        idx = self.city_keys.index(self.selected_city)
        slice_angle = 360 / len(self.city_keys)
        slice_center_cw = (idx * slice_angle) + (slice_angle / 2)
        target_rotation = slice_center_cw + 90 
        final_angle = target_rotation + (360 * random.randint(3, 5))
        
        duration = 4.0
        start = time.time()
        def animate():
            elapsed = time.time() - start
            if elapsed < duration:
                progress = elapsed / duration
                ease = 1 - (1 - progress)**3
                cur = final_angle * ease
                self._rotate_wheel(cur)
                self.after(16, animate)
            else:
                self._rotate_wheel(final_angle)
                MediaManager.stop_roll()
                self.open_popup_overlay() 
        animate()

    def open_popup_overlay(self):
        for widget in self.popup_frame.winfo_children():
            widget.destroy()

        hours = int(self.flight_time)
        minutes = int((self.flight_time - hours) * 60)
        time_text = f"Duration: {hours} Hr {minutes} Min" if minutes > 0 else f"Duration: {hours} Hours"

        ctk.CTkLabel(self.popup_frame, text="DESTINATION LOCKED", font=("Impact", 24)).pack(pady=(20,10))
        ctk.CTkLabel(self.popup_frame, text=self.selected_city.upper(), font=("Arial", 36, "bold"), text_color="#2ED573").pack()
        ctk.CTkLabel(self.popup_frame, text=time_text, font=("Arial", 18)).pack(pady=5)

        btn_row = ctk.CTkFrame(self.popup_frame, fg_color="transparent")
        btn_row.pack(pady=30)

        ctk.CTkButton(btn_row, text="❌ BACK", font=("Arial", 14, "bold"), 
                      height=40, width=120, fg_color="#555", hover_color="#333",
                      command=self.close_popup_overlay).pack(side="left", padx=10)

        ctk.CTkButton(btn_row, text="LOCK ME IN 🔒", font=("Arial", 14, "bold"), 
                      height=40, width=150, fg_color="#FF4757", hover_color="#D63030",
                      command=self.start_flight).pack(side="left", padx=10)

        self.popup_frame.place(relx=0.5, rely=0.5, anchor="center")
        self.popup_frame.lift() 

    def close_popup_overlay(self):
        self.popup_frame.place_forget()
        self.btn_spin.configure(state="normal")

    def open_logbook_overlay(self):
        for widget in self.logbook_frame.winfo_children():
            widget.destroy()

        logs = LogManager.get_logs()
        
        header = ctk.CTkFrame(self.logbook_frame, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(header, text="PILOT'S LOGBOOK", font=("Impact", 24)).pack(side="left")
        ctk.CTkButton(header, text="❌", width=30, fg_color="#555", hover_color="#333", command=self.close_logbook_overlay).pack(side="right")

        stats_frame = ctk.CTkFrame(self.logbook_frame)
        stats_frame.pack(fill="x", padx=10, pady=5)
        
        total_flights = len(logs)
        safe_flights = len([l for l in logs if l['status'] == 'LANDED'])
        total_hours = sum([l.get('duration', 0) for l in logs if l.get('status') == 'LANDED'])

        ctk.CTkLabel(stats_frame, text=f"FLIGHTS: {total_flights}", font=("Arial", 12, "bold")).pack(side="left", padx=20, pady=10)
        ctk.CTkLabel(stats_frame, text=f"HOURS: {total_hours:.1f}", font=("Arial", 12, "bold")).pack(side="left", padx=20)
        ctk.CTkLabel(stats_frame, text=f"SCORE: {safe_flights}/{total_flights}", font=("Arial", 12, "bold")).pack(side="right", padx=20)

        cols_frame = ctk.CTkFrame(self.logbook_frame, fg_color="transparent")
        cols_frame.pack(fill="x", padx=10, pady=(10,0))
        cols_frame.grid_columnconfigure(0, weight=1)
        cols_frame.grid_columnconfigure(1, weight=2)
        cols_frame.grid_columnconfigure(2, weight=1)
        cols_frame.grid_columnconfigure(3, weight=1)
        
        ctk.CTkLabel(cols_frame, text="DATE", font=("Arial", 14, "bold"), text_color="gray").grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(cols_frame, text="ROUTE", font=("Arial", 14, "bold"), text_color="gray").grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(cols_frame, text="TIME", font=("Arial", 14, "bold"), text_color="gray").grid(row=0, column=2, sticky="w")
        ctk.CTkLabel(cols_frame, text="STATUS", font=("Arial", 14, "bold"), text_color="gray").grid(row=0, column=3, sticky="e")

        scroll = ctk.CTkScrollableFrame(self.logbook_frame, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=10, pady=5)
        
        for entry in reversed(logs):
            row = ctk.CTkFrame(scroll)
            row.pack(fill="x", pady=2)
            row.grid_columnconfigure(0, weight=1)
            row.grid_columnconfigure(1, weight=2)
            row.grid_columnconfigure(2, weight=1)
            row.grid_columnconfigure(3, weight=1)
            
            color = "#2ED573" if entry['status'] == "LANDED" else "#FF4757"
            color = "#888" if entry['status'] == "ABORTED" else color
            status_text = entry['status']
            if entry['status'] == "LANDED": status_text = "✅ LANDED"
            elif entry['status'] == "CRASHED": status_text = "💥 CRASHED"
            elif entry['status'] == "ABORTED": status_text = "🛑 ABORTED"

            short_date = entry['date'][5:] 
            ctk.CTkLabel(row, text=short_date, font=("Arial", 12)).grid(row=0, column=0, sticky="w", padx=10, pady=10)
            dest_short = entry['destination'][:3].upper()
            ctk.CTkLabel(row, text=f"DEL ✈ {dest_short}", font=("Arial", 12, "bold")).grid(row=0, column=1, sticky="w", padx=5)
            ctk.CTkLabel(row, text=f"{entry.get('duration', 0)}h", font=("Arial", 12)).grid(row=0, column=2, sticky="w", padx=5)
            ctk.CTkLabel(row, text=status_text, text_color=color, font=("Arial", 12, "bold")).grid(row=0, column=3, sticky="e", padx=10)

        self.logbook_frame.place(relx=0.5, rely=0.5, anchor="center")
        self.logbook_frame.lift()

    def close_logbook_overlay(self):
        self.logbook_frame.place_forget()

    # ----------------------------------------
    # FLIGHT SCREEN
    # ----------------------------------------
    def _init_flight_screen(self):
        self.flight_frame = ctk.CTkFrame(self.container, fg_color="black")
        
        self.vid_lbl = ctk.CTkLabel(self.flight_frame, text="")
        self.vid_lbl.place(relx=0, rely=0, relwidth=1, relheight=1)
        
        self.pip_frame = ctk.CTkFrame(self.flight_frame, width=250, height=190, fg_color="#333", border_width=2, border_color="white")
        self.pip_frame.place(relx=0.02, rely=0.02, anchor="nw")
        self.pip_label = ctk.CTkLabel(self.pip_frame, text="Loading Cam...")
        self.pip_label.pack(expand=True, fill="both", padx=2, pady=2)

        self.timer_lbl = ctk.CTkLabel(self.flight_frame, text="00:00:00", font=("Courier", 50, "bold"), text_color="#00FF00", bg_color="black")
        self.timer_lbl.place(relx=0.9, rely=0.05, anchor="ne")
        
        self.crash_frame = ctk.CTkFrame(self.flight_frame, fg_color="red")
        self.crash_txt = ctk.CTkLabel(self.crash_frame, text="CRASH!", font=("Impact", 80), text_color="white")
        self.crash_txt.pack(expand=True)
        
        self.warn_frame = ctk.CTkFrame(self.flight_frame, fg_color="#FF8C00")
        self.warn_txt = ctk.CTkLabel(self.warn_frame, text="RETURN TO SEAT: 15", font=("Impact", 60), text_color="white")
        self.warn_txt.pack(expand=True)

        self.video_running = False

    def start_flight(self):
        self.close_popup_overlay() 
        self.attributes("-fullscreen", True)
        self.setup_frame.pack_forget()
        self.flight_frame.pack(fill="both", expand=True)
        
        MediaManager.play_music()
        
        if self.monitor: self.monitor.stop()
        if self.player: self.player.stop()
        
        self.monitor = Monitor(self.status_queue, self.pip_queue)
        self.player = VideoPlayer(self.bg_video_queue, VIDEO_PATH)
        
        self.monitor.start()
        self.player.start()
        
        self.video_running = True
        self.remaining = int(self.flight_time * 3600) 
        self.update_timer()

    def update_timer(self):
        if not self.video_running: return
        
        total_seconds = self.remaining
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        
        self.timer_lbl.configure(text=f"{h:02}:{m:02}:{s:02}")
        
        if self.remaining > 0:
            self.remaining -= 1
            self.after(1000, self.update_timer)
        else:
            self.success()

    def show_warning(self, data):
        # Data is (seconds, reason)
        seconds_left, reason = data
        self.warn_txt.configure(text=f"ALARM: {reason}!\n{seconds_left}s")
        MediaManager.start_alarm()
        if not self.is_warning_visible:
            self.warn_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.is_warning_visible = True

    def hide_warning(self):
        if self.is_warning_visible:
            self.warn_frame.place_forget()
            self.is_warning_visible = False
            MediaManager.stop_alarm()

    def show_crash(self, reason):
        self.hide_warning()
        self.crash_txt.configure(text=f"MAYDAY!\n{reason}")
        self.crash_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        MediaManager.stop_music()
        MediaManager.stop_alarm()
        MediaManager.play_sfx(ALARM_SOUND)
        
        elapsed = self.get_elapsed_hours()
        LogManager.save_trip(self.selected_city, elapsed, "CRASHED")
        
        self.video_running = False
        if self.monitor: self.monitor.stop()
        if self.player: self.player.stop()
        
        top = ctk.CTkToplevel(self)
        top.geometry("400x300")
        top.title("MISSION FAILED")
        top.attributes("-topmost", True)
        
        ctk.CTkLabel(top, text="YOU CRASHED!", font=("Impact", 30), text_color="#FF4757").pack(pady=(40, 20))
        ctk.CTkLabel(top, text=f"Reason: {reason}", font=("Arial", 16)).pack(pady=10)
        
        ctk.CTkButton(top, text="RETURN TO HOME 🏠", font=("Arial", 14, "bold"), 
                      fg_color="#555", height=50, width=200, 
                      command=lambda: [top.destroy(), self.return_to_home()]).pack(pady=30)

    def success(self):
        self.video_running = False
        if self.monitor: self.monitor.stop()
        if self.player: self.player.stop()
        
        LogManager.save_trip(self.selected_city, self.flight_time, "LANDED")
        
        MediaManager.stop_music()
        MediaManager.play_sfx(CHEER_SOUND)
        self.attributes("-fullscreen", False)
        
        top = ctk.CTkToplevel(self)
        top.geometry("400x300")
        top.title("MISSION ACCOMPLISHED")
        top.attributes("-topmost", True)
        
        ctk.CTkLabel(top, text="LANDED SAFELY!", font=("Impact", 30), text_color="#2ED573").pack(pady=(40, 20))
        ctk.CTkLabel(top, text=f"Welcome to {self.selected_city}", font=("Arial", 16)).pack(pady=10)
        
        ctk.CTkButton(top, text="RETURN TO HOME 🏠", font=("Arial", 14, "bold"), 
                      fg_color="#45B7D1", height=50, width=200, 
                      command=lambda: [top.destroy(), self.return_to_home()]).pack(pady=30)

    def on_close(self):
        if self.video_running:
             elapsed = self.get_elapsed_hours()
             LogManager.save_trip(self.selected_city, elapsed, "ABORTED")
        self.destroy()
        try: os._exit(0)
        except: pass

if __name__ == "__main__":
    app = FocusApp()
    app.mainloop()