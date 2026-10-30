
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import winsound
from datetime import datetime

import pyttsx3
import speech_recognition as sr


# ================ SETTINGS ================

EAR_THRESHOLD = 0.21
ALERT_SECONDS = 3
EMERGENCY_EYE_SECONDS = 8

MOUTH_OPEN_THRESHOLD = 0.35
YAWN_SECONDS = 1.2
YAWNS_FOR_BREAK = 3
YAWN_WINDOW_SECONDS = 60

# ==========================================

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 78
RIGHT_MOUTH = 308

alarm_active = False
consent_prompt_active = False
emergency_prompted = False

state_lock = threading.Lock()

dashboard_message = "Monitoring driver..."
consent_result = "Not requested"
emergency_status = "No alert requested"


# ================ SOUND ================

def play_drowsiness_alarm():
    while alarm_active:
        winsound.Beep(2200, 400)

        if alarm_active:
            winsound.Beep(1400, 400)


def play_yawn_beep():
    try:
        winsound.Beep(1000, 250)
    except RuntimeError:
        pass


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


def request_consent_once():
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

def eye_aspect_ratio(points):
    vertical1 = np.linalg.norm(points[1] - points[5])
    vertical2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal == 0:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


def get_mouth_ratio(face, width, height):
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


# ================ CAMERA ================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found. Check your webcam connection.")
    raise SystemExit

closed_since = None
yawn_since = None
yawn_counted = False
yawn_times = []

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

                def get_points(indices):
                    return np.array([
                        [
                            face.landmark[i].x * width,
                            face.landmark[i].y * height
                        ]
                        for i in indices
                    ], dtype=np.float32)

                left_eye = get_points(LEFT_EYE)
                right_eye = get_points(RIGHT_EYE)

                ear = (
                    eye_aspect_ratio(left_eye)
                    + eye_aspect_ratio(right_eye)
                ) / 2.0

                eyes_closed = ear < EAR_THRESHOLD

                for eye in (left_eye, right_eye):
                    for x, y in eye.astype(int):
                        cv2.circle(
                            frame, (x, y), 2, (0, 255, 0), -1
                        )

                (
                    mouth_ratio,
                    upper,
                    lower,
                    left_corner,
                    right_corner
                ) = get_mouth_ratio(face, width, height)

                for point in (
                    upper, lower, left_corner, right_corner
                ):
                    cv2.circle(
                        frame,
                        tuple(point.astype(int)),
                        3,
                        (255, 255, 0),
                        -1
                    )

            else:
                # Reset active measurements if face disappears
                closed_since = None
                alarm_active = False
                yawn_since = None
                yawn_counted = False

            # -------- EYE CLOSURE --------

            if face_detected and eyes_closed:
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

            elif face_detected:
                closed_since = None
                alarm_active = False

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

            # -------- COMPACT DASHBOARD --------

            # Draw a small translucent panel in the corner
            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (10, 10),
                (390, 235),
                (25, 25, 25),
                -1
            )

            cv2.addWeighted(
                overlay, 0.60,
                frame, 0.40,
                0, frame
            )

            cv2.putText(
                frame,
                "DRIVER SAFETY",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Eyes: {ear:.2f} | Closed: {eye_duration:.1f}s",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255) if eyes_closed else (0, 255, 0),
                1
            )

            cv2.putText(
                frame,
                f"Mouth ratio: {mouth_ratio:.2f}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1
            )

            cv2.putText(
                frame,
                f"Yawns (60s): {len(yawn_times)}",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                safety_status[:38],
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                status_color,
                1
            )

            # FIX: Define both variables before displaying them
            with state_lock:
                current_consent = consent_result
                current_message = dashboard_message

            cv2.putText(
                frame,
                "Voice: " + current_consent[:30],
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                current_message[:42],
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 200, 255),
                1
            )

            cv2.putText(
                frame,
                "Press Q to quit",
                (20, 222),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (200, 200, 200),
                1
            )

            if not face_detected:
                cv2.putText(
                    frame,
                    "Move into camera view",
                    (20, 265),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

            # Display exactly once per frame
            cv2.imshow("AI Driver Safety Dashboard", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    alarm_active = False
    camera.release()
    cv2.destroyAllWindows()
