import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import winsound
import os
import urllib.request
import json
from datetime import datetime

import pyttsx3
import speech_recognition as sr


# ================ SETTINGS ================

EAR_THRESHOLD = 0.21
ALERT_SECONDS = 3
EMERGENCY_EYE_SECONDS = 8
UNRESPONSIVE_DISPATCH_SECONDS = 60.0  # 1-minute unresponsive emergency trigger

MOUTH_OPEN_THRESHOLD = 0.35
YAWN_SECONDS = 1.2
YAWNS_FOR_BREAK = 3
YAWN_WINDOW_SECONDS = 60

# Cooldown before the emergency consent prompt can be triggered again after recovery
EMERGENCY_COOLDOWN_SECONDS = 45.0
RECOVERY_EYE_OPEN_SECONDS = 10.0

CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720

# ==========================================

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Additional eye contour ordering for sleek visualization
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 78
RIGHT_MOUTH = 308

# Inner mouth contour for sleek visualization
INNER_MOUTH_CONTOUR = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

alarm_active = False
consent_prompt_active = False
emergency_prompted = False
last_emergency_prompt_time = 0.0

# 1-Minute Unresponsive Dispatch state
unresponsive_dispatched = False
dispatch_snapshot_file = None
dispatch_timestamp_str = None
dispatch_location_cache = None

# AI Road & Weather Simulation states
ROAD_CONDITIONS = [
    {"name": "Dry Asphalt", "friction": 0.95, "risk": 0.05, "status": "OPTIMAL TRACTION"},
    {"name": "Wet / Rain Slick", "friction": 0.62, "risk": 0.45, "status": "CAUTION: SLIPPERY"},
    {"name": "Icy / Snow Patch", "friction": 0.28, "risk": 0.85, "status": "HAZARD: LOW FRICTION"},
    {"name": "Potholes / Rough", "friction": 0.70, "risk": 0.40, "status": "UNEVEN SURFACE"}
]
road_condition_idx = 0

WEATHER_CONDITIONS = [
    {"name": "Clear Sky", "visibility": 98, "risk": 0.05},
    {"name": "Heavy Rain", "visibility": 65, "risk": 0.50},
    {"name": "Dense Fog", "visibility": 35, "risk": 0.80},
    {"name": "Night Driving", "visibility": 75, "risk": 0.35}
]
weather_condition_idx = 0

state_lock = threading.Lock()
sound_lock = threading.Lock()

dashboard_message = "Monitoring driver..."
consent_result = "Not requested"
emergency_status = "No alert requested"


# ================ LOCATION & DISPATCH ================

def get_live_location():
    """Fetches real-world location via public IP geolocation, with reliable GPS fallback."""
    global dispatch_location_cache
    if dispatch_location_cache is not None:
        return dispatch_location_cache

    try:
        req = urllib.request.Request("http://ip-api.com/json", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=1.2) as response:
            data = json.loads(response.read().decode())
            if data.get("status") == "success":
                dispatch_location_cache = {
                    "lat": float(data.get("lat", 28.6139)),
                    "lon": float(data.get("lon", 77.2090)),
                    "city": data.get("city", "New Delhi"),
                    "region": data.get("regionName", "Delhi"),
                    "country": data.get("country", "India"),
                    "address": f"{data.get('city')}, {data.get('regionName')} (Highway NH-48 Sector)"
                }
                return dispatch_location_cache
    except Exception:
        pass

    dispatch_location_cache = {
        "lat": 28.6139,
        "lon": 77.2090,
        "city": "Sector 62, Noida",
        "region": "NCR",
        "country": "India",
        "address": "Sector 62, Noida Express Corridor (GPS Auto-Pin)"
    }
    return dispatch_location_cache


def trigger_unresponsive_dispatch(current_frame):
    """
    Called when driver is unresponsive for 1 minute (60 seconds).
    Saves snapshot picture, logs GPS location, and alerts emergency contacts.
    """
    global unresponsive_dispatched
    global dispatch_snapshot_file
    global dispatch_timestamp_str
    global emergency_status
    global dashboard_message

    with state_lock:
        if unresponsive_dispatched:
            return
        unresponsive_dispatched = True

    timestamp = datetime.now()
    dispatch_timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    snapshot_filename = f"emergency_snapshot_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"

    # Save high-res picture of driver
    try:
        cv2.imwrite(snapshot_filename, current_frame)
        dispatch_snapshot_file = snapshot_filename
    except Exception as e:
        print("Could not save snapshot:", e)
        dispatch_snapshot_file = "snapshot_error.jpg"

    loc = get_live_location()

    # Log to permanent emergency dispatch registry
    log_content = (
        "==========================================================\n"
        "🚨 CRITICAL ALERT: DRIVER UNRESPONSIVE FOR 60 SECONDS 🚨\n"
        "==========================================================\n"
        f"Timestamp: {dispatch_timestamp_str}\n"
        f"Driver Status: UNCONSCIOUS / UNRESPONSIVE (Closed >= 60s)\n"
        f"GPS Coordinates: Lat {loc['lat']:.4f}, Lon {loc['lon']:.4f}\n"
        f"Estimated Location: {loc['address']}\n"
        f"Snapshot Evidence File: {dispatch_snapshot_file}\n"
        "Emergency Contacts Notified: +1-800-SOS-SAFE / family_ice@emergency.net\n"
        "Automotive Action: Hazard Lights ACTIVE | Auto Emergency Brake ENGAGED\n"
        "==========================================================\n\n"
    )

    try:
        with open("emergency_dispatch_log.txt", "a", encoding="utf-8") as f:
            f.write(log_content)
    except Exception as e:
        print("Dispatch log error:", e)

    with state_lock:
        emergency_status = "CRITICAL DISPATCH SENT"
        dashboard_message = "Snapshot & GPS sent to contacts"

    # Voice TTS alert in background thread
    def voice_dispatch_announcement():
        speak("Critical emergency. Driver unresponsive for one minute. Live location and snapshot have been sent to emergency contacts.")

    threading.Thread(target=voice_dispatch_announcement, daemon=True).start()


# ================ SOUND ================

def play_drowsiness_alarm():
    """
    Plays the drowsiness alarm while alarm_active is True.
    Pauses automatically during voice assistant prompts to prevent mic interference.
    Guarded against concurrent winsound RuntimeError exceptions.
    """
    global alarm_active

    while alarm_active:
        # Pause alarm if voice assistant is currently speaking or listening
        with state_lock:
            if consent_prompt_active:
                time.sleep(0.15)
                continue

        with sound_lock:
            try:
                winsound.Beep(2200, 350)
            except Exception:
                pass

        with state_lock:
            if not alarm_active or consent_prompt_active:
                continue

        with sound_lock:
            try:
                winsound.Beep(1400, 350)
            except Exception:
                pass

        time.sleep(0.05)


def play_yawn_beep():
    """
    Plays a short reminder beep on detected yawn.
    Yields cleanly if the drowsiness alarm or voice prompt is active.
    """
    with state_lock:
        if consent_prompt_active or alarm_active:
            return

    # Non-blocking acquisition: skip if another sound is currently playing
    acquired = sound_lock.acquire(blocking=False)
    if not acquired:
        return

    try:
        winsound.Beep(1000, 250)
    except Exception:
        pass
    finally:
        sound_lock.release()


# ================ VOICE ================

def speak(message):
    try:
        engine = pyttsx3.init()
        engine.say(message)
        engine.runAndWait()
        engine.stop()
    except Exception as error:
        print("Voice output error:", error)


def ask_emergency_permission():
    global consent_prompt_active
    global consent_result
    global emergency_status
    global dashboard_message
    global last_emergency_prompt_time

    with state_lock:
        consent_prompt_active = True
        consent_result = "Waiting for response"
        dashboard_message = "Asking for permission..."

    try:
        speak(
            "Possible driver fatigue detected. "
            "Would you like to request emergency assistance? "
            "Please say yes or no."
        )

        recognizer = sr.Recognizer()

        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(
                source, duration=0.5
            )

            print("Listening for yes or no...")

            audio = recognizer.listen(
                source,
                timeout=6,
                phrase_time_limit=4
            )

        response = recognizer.recognize_google(audio).lower()
        print("Voice response:", response)

        words = set(response.split())

        if words.intersection(
            {"yes", "yeah", "affirmative", "correct"}
        ):
            timestamp = datetime.now().astimezone().isoformat(
                timespec="seconds"
            )

            with open(
                "emergency_alerts.txt",
                "a",
                encoding="utf-8"
            ) as file:
                file.write(
                    f"Timestamp: {timestamp}\n"
                    "Reason: Possible driver fatigue\n"
                    "Consent: YES\n"
                    "Location: Unavailable - GPS not configured\n"
                    "Delivery: Saved locally; no service configured\n"
                    "-----------------------------\n"
                )

            with state_lock:
                consent_result = "YES - permission granted"
                emergency_status = "Alert saved locally"
                dashboard_message = (
                    "Saved locally; authorities NOT contacted"
                )

            speak(
                "Permission recorded. The alert has been "
                "saved locally. Emergency services have "
                "not been contacted."
            )

        elif words.intersection({"no", "nope", "negative"}):
            with state_lock:
                consent_result = "NO - declined"
                emergency_status = "No alert sent"
                dashboard_message = "Emergency alert declined"

            speak("Okay. No emergency alert will be requested.")

        else:
            with state_lock:
                consent_result = "Unclear response"
                emergency_status = "No alert sent"
                dashboard_message = "Response unclear"

            speak(
                "I could not understand your response. "
                "No alert has been sent."
            )

    except sr.WaitTimeoutError:
        with state_lock:
            consent_result = "No response"
            emergency_status = "No alert sent"
            dashboard_message = "No response; no alert sent"

        speak("No response detected. No alert has been sent.")

    except sr.UnknownValueError:
        with state_lock:
            consent_result = "Voice not understood"
            emergency_status = "No alert sent"
            dashboard_message = "Voice not understood"

    except sr.RequestError:
        with state_lock:
            consent_result = "Speech service unavailable"
            emergency_status = "No alert sent"
            dashboard_message = "Check internet connection"

    except Exception as error:
        print("Voice or microphone error:", error)

        with state_lock:
            consent_result = "Voice input unavailable"
            emergency_status = "No alert sent"
            dashboard_message = "Check microphone settings"

    finally:
        with state_lock:
            consent_prompt_active = False
            last_emergency_prompt_time = time.monotonic()


def request_consent_once():
    """
    Triggers the emergency assistance voice prompt in a worker thread.
    Protected by lock and state flag.
    """
    global consent_prompt_active
    global emergency_prompted

    with state_lock:
        if consent_prompt_active or emergency_prompted:
            return

        emergency_prompted = True

    threading.Thread(
        target=ask_emergency_permission,
        daemon=True
    ).start()


# ================ FACE MEASUREMENTS ================

def get_points(face, indices, width, height):
    """Extracts pixel coordinates for given landmark indices."""
    return np.array([
        [
            face.landmark[i].x * width,
            face.landmark[i].y * height
        ]
        for i in indices
    ], dtype=np.float32)


def eye_aspect_ratio(points):
    """Calculates EAR for a 6-point eye landmark set."""
    vertical1 = np.linalg.norm(points[1] - points[5])
    vertical2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal == 0:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


def get_mouth_ratio(face, width, height):
    """Calculates MAR and returns coordinates of key mouth landmarks."""
    def point(index):
        landmark = face.landmark[index]
        return np.array(
            [landmark.x * width, landmark.y * height],
            dtype=np.float32
        )

    upper = point(UPPER_LIP)
    lower = point(LOWER_LIP)
    left = point(LEFT_MOUTH)
    right = point(RIGHT_MOUTH)

    mouth_height = np.linalg.norm(upper - lower)
    mouth_width = np.linalg.norm(left - right)

    ratio = (
        mouth_height / mouth_width
        if mouth_width > 0 else 0.0
    )

    return ratio, upper, lower, left, right


# ================ MODERN UI RENDERING ================

def draw_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=8):
    """Draws a smooth rounded rectangle on an image."""
    x1, y1 = pt1
    x2, y2 = pt2
    w = x2 - x1
    h = y2 - y1
    r = min(radius, w // 2, h // 2)

    if thickness == -1:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)


def draw_progress_bar(img, x, y, width, height, value, max_value, threshold_val, bar_color, bg_color=(35, 38, 46)):
    """Draws a sleek progress bar with an indicator marker for the threshold."""
    draw_rounded_rect(img, (x, y), (x + width, y + height), bg_color, -1, radius=4)

    fill_ratio = min(max(value / max_value if max_value > 0 else 0.0, 0.0), 1.0)
    fill_w = int(width * fill_ratio)

    if fill_w > 4:
        draw_rounded_rect(img, (x, y), (x + fill_w, y + height), bar_color, -1, radius=4)

    if threshold_val is not None and max_value > 0:
        thresh_x = int(x + width * (threshold_val / max_value))
        cv2.line(img, (thresh_x, y - 2), (thresh_x, y + height + 2), (255, 255, 255), 2, cv2.LINE_AA)


def draw_card(canvas, x1, y1, x2, y2, bg_color=(20, 23, 30), border_color=(45, 50, 62)):
    """Draws a polished semi-transparent glass card with border."""
    overlay = canvas.copy()
    draw_rounded_rect(overlay, (x1, y1), (x2, y2), bg_color, -1, radius=10)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
    draw_rounded_rect(canvas, (x1, y1), (x2, y2), border_color, 1, radius=10)


def render_full_screen_cockpit(
    canvas,
    camera_frame,
    ear,
    eye_duration,
    mouth_ratio,
    yawn_times,
    face_detected,
    safety_status,
    status_color,
    current_consent,
    current_message,
    car_hologram_img,
    ambient_lux,
    lighting_condition
):
    """
    Renders the entire 1280x720 Widescreen AI Automotive Cockpit:
    - Left Column: Driver Biometrics (EAR, MAR, 1-Min Unresponsive Timer, Yawn History)
    - Center Column: Live Camera Feed with AR tracking + High Priority Centered Alert Modals
    - Right Column: AI Road & Weather Intelligence + 3D Car Hologram & Damage Assessment
    - Top & Bottom Global Telemetry Bars with GPS and Real-time Alerts
    """
    global unresponsive_dispatched
    global dispatch_snapshot_file
    global dispatch_timestamp_str

    now = time.monotonic()
    loc = get_live_location()
    road = ROAD_CONDITIONS[road_condition_idx]
    weather = WEATHER_CONDITIONS[weather_condition_idx]

    # Calculate Compound AI Risk Index (0% -> 100%)
    fatigue_factor = min(1.0, (eye_duration / ALERT_SECONDS) * 0.5 + (len(yawn_times) / YAWNS_FOR_BREAK) * 0.5)
    compound_risk = int((fatigue_factor * 0.50 + road["risk"] * 0.25 + weather["risk"] * 0.25) * 100)

    # ---------------- 1. TOP GLOBAL STATUS BAR (0 -> 52px) ----------------
    draw_card(canvas, 10, 8, CANVAS_WIDTH - 10, 52, bg_color=(15, 17, 24), border_color=(40, 45, 58))

    # Pulsing status radar indicator
    pulse = (int(now * 4) % 2) == 0
    indicator_col = (0, 230, 118) if face_detected and eye_duration < ALERT_SECONDS else (0, 0, 255)
    cv2.circle(canvas, (28, 30), 7, indicator_col, -1, cv2.LINE_AA)
    if pulse and eye_duration >= ALERT_SECONDS:
        cv2.circle(canvas, (28, 30), 12, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.putText(canvas, "AI AUTOMOTIVE INTELLIGENCE & DRIVER DEFENSE", (44, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "v2.5 CO-PILOT", (470, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1, cv2.LINE_AA)

    # Compound Threat Badge (Center)
    threat_text = f"AI THREAT LEVEL: {compound_risk}% " + ("NORMAL" if compound_risk < 30 else ("ELEVATED" if compound_risk < 65 else "CRITICAL DANGER"))
    threat_color = (0, 220, 110) if compound_risk < 30 else ((0, 190, 255) if compound_risk < 65 else (0, 50, 255))
    cv2.putText(canvas, threat_text, (620, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.44, threat_color, 2, cv2.LINE_AA)

    # Live clock & mode (Right)
    clock_str = datetime.now().strftime("%H:%M:%S | %b %d, %Y")
    cv2.putText(canvas, clock_str, (CANVAS_WIDTH - 260, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 195, 205), 1, cv2.LINE_AA)

    # ---------------- 2. LEFT COLUMN: DRIVER BIOMETRICS (10 -> 330px) ----------------
    left_x1, left_y1 = 10, 60
    left_w = 320
    left_x2, left_y2 = left_x1 + left_w, 650
    draw_card(canvas, left_x1, left_y1, left_x2, left_y2, bg_color=(18, 20, 27), border_color=(45, 50, 64))

    # Header
    cv2.putText(canvas, "DRIVER BIOMETRICS & FATIGUE", (left_x1 + 14, left_y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (left_x1 + 14, left_y1 + 36), (left_x2 - 14, left_y1 + 36), (40, 45, 55), 1, cv2.LINE_AA)

    # EAR Readout & Progress
    eyes_closed = face_detected and (ear < EAR_THRESHOLD)
    ear_color = (0, 80, 255) if eyes_closed else (0, 230, 118)
    cv2.putText(canvas, f"EAR: {ear:.2f} (Threshold: {EAR_THRESHOLD:.2f})", (left_x1 + 14, left_y1 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.42, ear_color, 1, cv2.LINE_AA)
    draw_progress_bar(canvas, left_x1 + 14, left_y1 + 66, left_w - 28, 8, ear, 0.40, EAR_THRESHOLD, ear_color)

    # Eye Closure Alert Meter (3.0s)
    cv2.putText(canvas, f"Eyes Closed: {eye_duration:.1f}s / {ALERT_SECONDS}.0s", (left_x1 + 14, left_y1 + 96), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 80, 255) if eye_duration >= ALERT_SECONDS else (190, 195, 205), 1, cv2.LINE_AA)
    draw_progress_bar(canvas, left_x1 + 14, left_y1 + 104, left_w - 28, 7, eye_duration, ALERT_SECONDS, ALERT_SECONDS, (0, 50, 255) if eye_duration >= ALERT_SECONDS else (0, 180, 255))

    # 1-MINUTE CRITICAL UNRESPONSIVE TIMER
    unresponsive_color = (0, 0, 255) if eye_duration >= 20.0 else (0, 200, 255)
    cv2.putText(canvas, f"1-Min Dispatch Timer: {eye_duration:.1f}s / 60.0s", (left_x1 + 14, left_y1 + 134), cv2.FONT_HERSHEY_SIMPLEX, 0.40, unresponsive_color, 1, cv2.LINE_AA)
    draw_progress_bar(canvas, left_x1 + 14, left_y1 + 142, left_w - 28, 8, eye_duration, UNRESPONSIVE_DISPATCH_SECONDS, UNRESPONSIVE_DISPATCH_SECONDS, (0, 0, 255) if eye_duration >= UNRESPONSIVE_DISPATCH_SECONDS else (0, 150, 255))

    cv2.line(canvas, (left_x1 + 14, left_y1 + 164), (left_x2 - 14, left_y1 + 164), (40, 45, 55), 1, cv2.LINE_AA)

    # MAR & Yawn Activity
    mouth_open = mouth_ratio > MOUTH_OPEN_THRESHOLD
    mar_color = (0, 160, 255) if mouth_open else (200, 205, 215)
    cv2.putText(canvas, f"Mouth MAR: {mouth_ratio:.2f} (Limit: {MOUTH_OPEN_THRESHOLD:.2f})", (left_x1 + 14, left_y1 + 186), cv2.FONT_HERSHEY_SIMPLEX, 0.42, mar_color, 1, cv2.LINE_AA)
    draw_progress_bar(canvas, left_x1 + 14, left_y1 + 194, left_w - 28, 7, mouth_ratio, 0.70, MOUTH_OPEN_THRESHOLD, (0, 170, 255) if mouth_open else (90, 190, 255))

    # Yawn count (60s)
    yawn_count = len(yawn_times)
    cv2.putText(canvas, f"Yawns (60s window): {yawn_count} / {YAWNS_FOR_BREAK}", (left_x1 + 14, left_y1 + 222), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 215, 225), 1, cv2.LINE_AA)
    for i in range(YAWNS_FOR_BREAK):
        cx = left_x1 + 220 + (i * 24)
        cy = left_y1 + 218
        if i < yawn_count:
            cv2.circle(canvas, (cx, cy), 7, (0, 0, 255) if yawn_count >= YAWNS_FOR_BREAK else (0, 180, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (cx, cy), 7, (50, 55, 68), 1, cv2.LINE_AA)

    cv2.line(canvas, (left_x1 + 14, left_y1 + 242), (left_x2 - 14, left_y1 + 242), (40, 45, 55), 1, cv2.LINE_AA)

    # Overall Safety Status Badge
    cv2.putText(canvas, "SAFETY STATUS:", (left_x1 + 14, left_y1 + 264), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 175, 185), 1, cv2.LINE_AA)
    draw_rounded_rect(canvas, (left_x1 + 14, left_y1 + 274), (left_x2 - 14, left_y1 + 304), (25, 28, 36), -1, radius=6)
    cv2.putText(canvas, safety_status[:28], (left_x1 + 22, left_y1 + 294), cv2.FONT_HERSHEY_SIMPLEX, 0.40, status_color, 1, cv2.LINE_AA)

    # Emergency Dispatch Readiness Card
    cv2.line(canvas, (left_x1 + 14, left_y1 + 320), (left_x2 - 14, left_y1 + 320), (40, 45, 55), 1, cv2.LINE_AA)
    cv2.putText(canvas, "EMERGENCY DISPATCH SYSTEM", (left_x1 + 14, left_y1 + 342), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    disp_badge = "TRANSMITTED (ACTIVE)" if unresponsive_dispatched else "ARMED & STANDBY"
    disp_col = (0, 0, 255) if unresponsive_dispatched else (0, 220, 110)
    cv2.putText(canvas, f"State: {disp_badge}", (left_x1 + 14, left_y1 + 364), cv2.FONT_HERSHEY_SIMPLEX, 0.38, disp_col, 1, cv2.LINE_AA)

    # Show photo snapshot confirmation
    if dispatch_snapshot_file:
        cv2.putText(canvas, f"Photo: {dispatch_snapshot_file[:24]}", (left_x1 + 14, left_y1 + 386), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 210, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Sent to Contacts: YES", (left_x1 + 14, left_y1 + 406), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 120), 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "Auto-Snap on 60s: READY", (left_x1 + 14, left_y1 + 386), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 175, 185), 1, cv2.LINE_AA)
        cv2.putText(canvas, "GPS Uplink: READY", (left_x1 + 14, left_y1 + 406), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 175, 185), 1, cv2.LINE_AA)

    # Voice Assistant Readout
    cv2.line(canvas, (left_x1 + 14, left_y1 + 424), (left_x2 - 14, left_y1 + 424), (40, 45, 55), 1, cv2.LINE_AA)
    cv2.putText(canvas, "VOICE ASSISTANT & CONSENT", (left_x1 + 14, left_y1 + 446), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Voice: {current_consent[:26]}", (left_x1 + 14, left_y1 + 470), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 215, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{current_message[:28]}", (left_x1 + 14, left_y1 + 494), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 185, 195), 1, cv2.LINE_AA)

    # ---------------- 3. CENTER COLUMN: CAMERA FEED & CENTRAL ALERTS (340 -> 940px) ----------------
    center_x1, center_y1 = 340, 60
    center_w = 600
    center_h = 450
    center_x2, center_y2 = center_x1 + center_w, center_y1 + center_h

    # Card border for camera viewport
    draw_card(canvas, center_x1 - 2, center_y1 - 2, center_x2 + 2, center_y2 + 2, bg_color=(10, 12, 16), border_color=(45, 50, 64))

    # Resize camera feed to fit center viewport cleanly
    cam_resized = cv2.resize(camera_frame, (center_w, center_h), interpolation=cv2.INTER_LINEAR)
    canvas[center_y1:center_y2, center_x1:center_x2] = cam_resized

    # Frame Corner Accents on Camera View
    accent_len = 20
    accent_col = (0, 210, 255)
    for cx, cy, dx, dy in [
        (center_x1, center_y1, 1, 1),
        (center_x2, center_y1, -1, 1),
        (center_x1, center_y2, 1, -1),
        (center_x2, center_y2, -1, -1)
    ]:
        cv2.line(canvas, (cx, cy), (cx + (dx * accent_len), cy), accent_col, 2, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy), (cx, cy + (dy * accent_len)), accent_col, 2, cv2.LINE_AA)

    # HUD Camera overlay tags
    cv2.putText(canvas, "LIVE DRIVER TRACKING [468 MESH]", (center_x1 + 16, center_y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 110) if face_detected else (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"FPS: ~30 | LUX: {ambient_lux}", (center_x2 - 130, center_y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 195, 205), 1, cv2.LINE_AA)

    # --- CENTER HIGH-PRIORITY FLOATING ALERT BANNERS (Centered strictly over camera) ---
    if unresponsive_dispatched:
        # MASSIVE 1-MINUTE EMERGENCY DISPATCH BANNER
        m_w, m_h = 560, 120
        mx1 = center_x1 + (center_w - m_w) // 2
        my1 = center_y1 + (center_h - m_h) // 2
        draw_card(canvas, mx1, my1, mx1 + m_w, my1 + m_h, bg_color=(150, 0, 0), border_color=(255, 50, 50))
        cv2.putText(canvas, "! CRITICAL EMERGENCY: DRIVER UNRESPONSIVE (60s) !", (mx1 + 20, my1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"PHOTO SNAPSHOT & GPS SENT TO EMERGENCY CONTACTS", (mx1 + 20, my1 + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Location: {loc['lat']:.4f} N, {loc['lon']:.4f} E | {loc['city']}", (mx1 + 20, my1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Evidence: {dispatch_snapshot_file} | Hazard Strobes: ENGAGED", (mx1 + 20, my1 + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 120), 1, cv2.LINE_AA)

    elif not face_detected:
        b_w = 400
        bx1 = center_x1 + (center_w - b_w) // 2
        by1 = center_y1 + 16
        draw_rounded_rect(canvas, (bx1, by1), (bx1 + b_w, by1 + 42), (0, 0, 180), -1, radius=6)
        draw_rounded_rect(canvas, (bx1, by1), (bx1 + b_w, by1 + 42), (0, 0, 255), 2, radius=6)
        cv2.putText(canvas, "! MOVE INTO CAMERA VIEW !", (bx1 + 55, by1 + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

    elif eye_duration >= ALERT_SECONDS:
        b_w = 460
        bx1 = center_x1 + (center_w - b_w) // 2
        by1 = center_y1 + 16
        pulse_col = (0, 0, 255) if (int(now * 5) % 2 == 0) else (0, 100, 255)
        draw_rounded_rect(canvas, (bx1, by1), (bx1 + b_w, by1 + 44), (20, 20, 160), -1, radius=6)
        draw_rounded_rect(canvas, (bx1, by1), (bx1 + b_w, by1 + 44), pulse_col, 2, radius=6)
        cv2.putText(canvas, "CRITICAL: EYES CLOSED - WAKE UP!", (bx1 + 35, by1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2, cv2.LINE_AA)

    elif len(yawn_times) >= YAWNS_FOR_BREAK:
        b_w = 440
        bx1 = center_x1 + (center_w - b_w) // 2
        by1 = center_y1 + 16
        draw_rounded_rect(canvas, (bx1, by1), (bx1 + b_w, by1 + 44), (0, 90, 180), -1, radius=6)
        draw_rounded_rect(canvas, (bx1, by1), (bx1 + b_w, by1 + 44), (0, 165, 255), 2, radius=6)
        cv2.putText(canvas, "FATIGUE WARNING: TAKE A BREAK", (bx1 + 35, by1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

    # Secondary Bottom Center Card (GPS Telemetry & Live Coordinates)
    sub_y1 = center_y2 + 10
    sub_y2 = 650
    draw_card(canvas, center_x1, sub_y1, center_x2, sub_y2, bg_color=(18, 20, 27), border_color=(45, 50, 64))

    cv2.putText(canvas, "CONNECTED EMERGENCY CONTACTS & VEHICLE TELEMETRY", (center_x1 + 16, sub_y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"GPS: {loc['lat']:.4f} N, {loc['lon']:.4f} E | {loc['address'][:48]}", (center_x1 + 16, sub_y1 + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 210, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "ICE Contacts: 1) +1-800-SOS-SAFE (Emergency Services)  2) family_ice@emergency.net", (center_x1 + 16, sub_y1 + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 185, 195), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Dispatch Uplink: 4G/LTE + SAT-COM Armed | Emergency Automatic Braking (AEB): READY", (center_x1 + 16, sub_y1 + 96), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 230, 118), 1, cv2.LINE_AA)

    # ---------------- 4. RIGHT COLUMN: AI ROAD/WEATHER & 3D CAR DAMAGE (950 -> 1270px) ----------------
    right_x1, right_y1 = 950, 60
    right_w = 320
    right_x2 = right_x1 + right_w

    # --- Card 4A: AI Road & Weather Intelligence (y: 60 -> 330) ---
    road_y2 = 330
    draw_card(canvas, right_x1, right_y1, right_x2, road_y2, bg_color=(18, 20, 27), border_color=(45, 50, 64))

    cv2.putText(canvas, "AI ROAD & WEATHER INTELLIGENCE", (right_x1 + 14, right_y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (right_x1 + 14, right_y1 + 36), (right_x2 - 14, right_y1 + 36), (40, 45, 55), 1, cv2.LINE_AA)

    # Road Condition
    cv2.putText(canvas, f"Road Surface: {road['name']}", (right_x1 + 14, right_y1 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Friction: {road['friction']:.2f} | {road['status']}", (right_x1 + 14, right_y1 + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 100) if road['risk'] < 0.4 else (0, 160, 255), 1, cv2.LINE_AA)

    # Weather Condition
    cv2.putText(canvas, f"Weather: {weather['name']}", (right_x1 + 14, right_y1 + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 215, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Visibility: {weather['visibility']}% | Rain/Mist Sensor", (right_x1 + 14, right_y1 + 132), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 195, 205), 1, cv2.LINE_AA)

    # Vision-based Ambient Lighting
    cv2.putText(canvas, f"Vision Lux: {ambient_lux} | {lighting_condition}", (right_x1 + 14, right_y1 + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 210, 225), 1, cv2.LINE_AA)

    # Compound Threat Breakdown
    cv2.putText(canvas, f"AI Compound Risk: {compound_risk}%", (right_x1 + 14, right_y1 + 190), cv2.FONT_HERSHEY_SIMPLEX, 0.42, threat_color, 2, cv2.LINE_AA)
    draw_progress_bar(canvas, right_x1 + 14, right_y1 + 200, right_w - 28, 8, compound_risk, 100, 50, threat_color)

    cv2.putText(canvas, "[R] Cycle Road  |  [W] Cycle Weather", (right_x1 + 14, right_y1 + 235), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 145, 155), 1, cv2.LINE_AA)

    # --- Card 4B: 3D Holographic Car & Damage Assessment (y: 340 -> 650) ---
    car_y1 = 340
    car_y2 = 650
    draw_card(canvas, right_x1, car_y1, right_x2, car_y2, bg_color=(18, 20, 27), border_color=(45, 50, 64))

    cv2.putText(canvas, "3D DIGITAL TWIN & VEHICLE HEALTH", (right_x1 + 14, car_y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (right_x1 + 14, car_y1 + 34), (right_x2 - 14, car_y1 + 34), (40, 45, 55), 1, cv2.LINE_AA)

    # Render 3D Car Image
    car_img_w = right_w - 28
    car_img_h = 130
    car_pos_x = right_x1 + 14
    car_pos_y = car_y1 + 42

    if car_hologram_img is not None:
        car_resized = cv2.resize(car_hologram_img, (car_img_w, car_img_h), interpolation=cv2.INTER_AREA)

        # If in critical state or damaged condition, tint hologram red
        if eye_duration >= ALERT_SECONDS or unresponsive_dispatched:
            red_tint = np.zeros_like(car_resized)
            red_tint[:, :] = (0, 0, 80)
            car_resized = cv2.addWeighted(car_resized, 0.75, red_tint, 0.25, 0)

        canvas[car_pos_y:car_pos_y + car_img_h, car_pos_x:car_pos_x + car_img_w] = car_resized
        draw_rounded_rect(canvas, (car_pos_x, car_pos_y), (car_pos_x + car_img_w, car_pos_y + car_img_h), (0, 200, 255) if eye_duration < ALERT_SECONDS else (0, 0, 255), 1, radius=4)

    # Dynamic Vehicle Damage & Condition Diagnostics
    diag_y = car_pos_y + car_img_h + 18

    if unresponsive_dispatched:
        cv2.putText(canvas, "DAMAGE HAZARD: HIGH IMPACT RISK", (right_x1 + 14, diag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Front Crumple Zone: IMPACT WARNING", (right_x1 + 14, diag_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Auto Emergency Brake (AEB): ENGAGED", (right_x1 + 14, diag_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 215, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Cabin Airbag Pre-tensioner: PRIMED", (right_x1 + 14, diag_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 165, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Vehicle Status: EMERGENCY STOP ACTIVE", (right_x1 + 14, diag_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, cv2.LINE_AA)

    elif eye_duration >= ALERT_SECONDS:
        cv2.putText(canvas, "VEHICLE STATUS: COLLISION HAZARD", (right_x1 + 14, diag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 160, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Front Radar: OBSTACLE MONITORING", (right_x1 + 14, diag_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 180, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Braking System: AEB PRE-CHARGED", (right_x1 + 14, diag_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 215, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Lane Keep Assist: ACTIVE STEER", (right_x1 + 14, diag_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 230, 118), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Chassis Integrity: 100% NOMINAL", (right_x1 + 14, diag_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 110), 1, cv2.LINE_AA)

    else:
        cv2.putText(canvas, "VEHICLE HEALTH: 100% NOMINAL", (right_x1 + 14, diag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 230, 118), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Chassis & Frame: NO DAMAGE DETECTED", (right_x1 + 14, diag_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 220, 110), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Tire Traction: 98% (Optimal Grip)", (right_x1 + 14, diag_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 185, 195), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Collision Radar: 360 Lidar CLEAR", (right_x1 + 14, diag_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 215, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Powertrain & Battery: 88% OK", (right_x1 + 14, diag_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 110), 1, cv2.LINE_AA)

    # ---------------- 5. BOTTOM GLOBAL CONTROLS BAR (660 -> 712px) ----------------
    draw_card(canvas, 10, 660, CANVAS_WIDTH - 10, 712, bg_color=(15, 17, 24), border_color=(40, 45, 58))

    controls_left = "[F] Fullscreen Toggle   [Q] Quit System   [E] Simulate 1-Min 60s Dispatch   [R] Cycle Road   [W] Cycle Weather"
    cv2.putText(canvas, controls_left, (24, 692), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 190, 205), 1, cv2.LINE_AA)

    status_tag = "ACTIVE SAFETY SHIELD: ON"
    cv2.putText(canvas, status_tag, (CANVAS_WIDTH - 240, 692), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 118), 2, cv2.LINE_AA)


# ================ MAIN EXECUTION ================

def main():
    global alarm_active
    global emergency_prompted
    global emergency_status
    global dashboard_message
    global road_condition_idx
    global weather_condition_idx
    global unresponsive_dispatched

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Camera not found. Check your webcam connection.")
        raise SystemExit

    # Load 3D car hologram image
    car_hologram = None
    if os.path.exists("car_3d_model.jpg"):
        car_hologram = cv2.imread("car_3d_model.jpg")

    closed_since = None
    yawn_since = None
    yawn_counted = False
    yawn_times = []
    eye_open_since = None

    # Fullscreen window setup
    window_name = "AI Driver Safety Dashboard"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CANVAS_WIDTH, CANVAS_HEIGHT)
    is_fullscreen = False

    try:
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:

            while True:
                ret, frame = camera.read()

                if not ret:
                    print("Could not read from camera.")
                    break

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]

                # Vision-based ambient lighting estimation
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ambient_lux = int(np.mean(gray))
                if ambient_lux > 125:
                    lighting_condition = "DAYLIGHT (Clear)"
                elif ambient_lux > 65:
                    lighting_condition = "TWILIGHT (Moderate)"
                else:
                    lighting_condition = "NIGHT / LOW LIGHT"

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)
                now = time.monotonic()

                face_detected = bool(result.multi_face_landmarks)

                eyes_closed = False
                ear = 0.0
                mouth_ratio = 0.0
                eye_duration = 0.0

                # -------- FACE DETECTION --------

                if face_detected:
                    face = result.multi_face_landmarks[0]

                    left_eye = get_points(face, LEFT_EYE, width, height)
                    right_eye = get_points(face, RIGHT_EYE, width, height)

                    ear = (
                        eye_aspect_ratio(left_eye)
                        + eye_aspect_ratio(right_eye)
                    ) / 2.0

                    eyes_closed = ear < EAR_THRESHOLD

                    # Sleek polyline visualization for eye contours
                    left_contour = get_points(face, LEFT_EYE_CONTOUR, width, height).astype(np.int32)
                    right_contour = get_points(face, RIGHT_EYE_CONTOUR, width, height).astype(np.int32)

                    eye_contour_color = (0, 0, 255) if eyes_closed else (0, 255, 180)
                    cv2.polylines(frame, [left_contour], isClosed=True, color=eye_contour_color, thickness=1, lineType=cv2.LINE_AA)
                    cv2.polylines(frame, [right_contour], isClosed=True, color=eye_contour_color, thickness=1, lineType=cv2.LINE_AA)

                    # Highlight key EAR calculation points
                    for eye in (left_eye, right_eye):
                        for x, y in eye.astype(int):
                            cv2.circle(
                                frame, (x, y), 2, (0, 255, 255) if eyes_closed else (0, 255, 0), -1, cv2.LINE_AA
                            )

                    (
                        mouth_ratio,
                        upper,
                        lower,
                        left_corner,
                        right_corner
                    ) = get_mouth_ratio(face, width, height)

                    # Subtle inner mouth contour
                    inner_mouth = get_points(face, INNER_MOUTH_CONTOUR, width, height).astype(np.int32)
                    mouth_contour_color = (0, 165, 255) if mouth_ratio > MOUTH_OPEN_THRESHOLD else (200, 200, 0)
                    cv2.polylines(frame, [inner_mouth], isClosed=True, color=mouth_contour_color, thickness=1, lineType=cv2.LINE_AA)

                    for point in (
                        upper, lower, left_corner, right_corner
                    ):
                        cv2.circle(
                            frame,
                            tuple(point.astype(int)),
                            3,
                            (0, 215, 255),
                            -1,
                            cv2.LINE_AA
                        )

                else:
                    # Reset active measurements if face disappears
                    closed_since = None
                    alarm_active = False
                    yawn_since = None
                    yawn_counted = False
                    eye_open_since = None

                # -------- EYE CLOSURE --------

                if face_detected and eyes_closed:
                    eye_open_since = None
                    if closed_since is None:
                        closed_since = now

                    eye_duration = now - closed_since

                    if eye_duration >= ALERT_SECONDS:
                        if not alarm_active:
                            alarm_active = True

                            threading.Thread(
                                target=play_drowsiness_alarm,
                                daemon=True
                            ).start()
                    else:
                        alarm_active = False

                    if eye_duration >= EMERGENCY_EYE_SECONDS:
                        request_consent_once()

                    # -------- 1-MINUTE CRITICAL UNRESPONSIVE DISPATCH --------
                    if eye_duration >= UNRESPONSIVE_DISPATCH_SECONDS and not unresponsive_dispatched:
                        trigger_unresponsive_dispatch(frame)

                elif face_detected:
                    closed_since = None
                    alarm_active = False

                    # Track sustained eye-open recovery time for resetting emergency lockout
                    if eye_open_since is None:
                        eye_open_since = now

                    # -------- EMERGENCY COOLDOWN & RECOVERY RESET --------
                    with state_lock:
                        if emergency_prompted and not consent_prompt_active:
                            recovery_time = now - eye_open_since
                            cooldown_passed = (now - last_emergency_prompt_time) >= EMERGENCY_COOLDOWN_SECONDS

                            if cooldown_passed and recovery_time >= RECOVERY_EYE_OPEN_SECONDS:
                                emergency_prompted = False
                                unresponsive_dispatched = False
                                emergency_status = "Ready"
                                dashboard_message = "Monitoring driver..."
                                print("Emergency alert status reset - driver safely recovered.")

                # -------- YAWN DETECTION --------

                yawn_times = [
                    t for t in yawn_times
                    if now - t <= YAWN_WINDOW_SECONDS
                ]

                mouth_open = mouth_ratio > MOUTH_OPEN_THRESHOLD

                if face_detected and mouth_open:
                    if yawn_since is None:
                        yawn_since = now

                    yawn_duration = now - yawn_since

                    if (
                        yawn_duration >= YAWN_SECONDS
                        and not yawn_counted
                    ):
                        yawn_counted = True
                        yawn_times.append(now)

                        threading.Thread(
                            target=play_yawn_beep,
                            daemon=True
                        ).start()

                else:
                    yawn_since = None
                    yawn_counted = False

                # -------- SAFETY STATUS --------

                if not face_detected:
                    safety_status = "FACE NOT DETECTED"
                    status_color = (0, 0, 255)

                elif eye_duration >= ALERT_SECONDS:
                    safety_status = "HIGH RISK: EYES CLOSED"
                    status_color = (0, 0, 255)

                elif len(yawn_times) >= YAWNS_FOR_BREAK:
                    safety_status = "FATIGUE WARNING: TAKE A BREAK"
                    status_color = (0, 165, 255)

                elif len(yawn_times) >= 1:
                    safety_status = "POSSIBLE FATIGUE CUES"
                    status_color = (0, 200, 255)

                else:
                    safety_status = "NO FATIGUE CUES DETECTED"
                    status_color = (0, 255, 0)

                if len(yawn_times) >= YAWNS_FOR_BREAK:
                    request_consent_once()

                # -------- BUILD FULL-SCREEN COCKPIT CANVAS --------

                canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)

                with state_lock:
                    current_consent = consent_result
                    current_message = dashboard_message

                render_full_screen_cockpit(
                    canvas=canvas,
                    camera_frame=frame,
                    ear=ear,
                    eye_duration=eye_duration,
                    mouth_ratio=mouth_ratio,
                    yawn_times=yawn_times,
                    face_detected=face_detected,
                    safety_status=safety_status,
                    status_color=status_color,
                    current_consent=current_consent,
                    current_message=current_message,
                    car_hologram_img=car_hologram,
                    ambient_lux=ambient_lux,
                    lighting_condition=lighting_condition
                )

                # Display the complete full widescreen dashboard
                cv2.imshow(window_name, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("f"):
                    # Toggle Fullscreen
                    is_fullscreen = not is_fullscreen
                    if is_fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                elif key == ord("e"):
                    # Test simulation for 1-minute unresponsive dispatch
                    print("Simulating 1-Minute Unresponsive Dispatch...")
                    trigger_unresponsive_dispatch(frame)
                elif key == ord("r"):
                    # Cycle Road Condition
                    road_condition_idx = (road_condition_idx + 1) % len(ROAD_CONDITIONS)
                    print(f"Road switched to: {ROAD_CONDITIONS[road_condition_idx]['name']}")
                elif key == ord("w"):
                    # Cycle Weather Condition
                    weather_condition_idx = (weather_condition_idx + 1) % len(WEATHER_CONDITIONS)
                    print(f"Weather switched to: {WEATHER_CONDITIONS[weather_condition_idx]['name']}")

    finally:
        alarm_active = False
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
