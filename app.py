import cv2
import easyocr
import pyttsx3
import os
import time
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
MODEL_PATH = r"C:\app\yolov8n.pt"
LITE_MODE = True          # وضع توفير الطاقة
SCAN_INTERVAL = 3         # كل كام ثانية يعمل تحليل

# =========================
# CHECK MODEL
# =========================
if not os.path.exists(MODEL_PATH):
    print("Model not found! Check path:", MODEL_PATH)
    exit()

model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=False)

# =========================
# TEXT TO SPEECH
# =========================
engine = pyttsx3.init()

def speak(text):
    print("AI:", text)
    engine.say(text)
    engine.runAndWait()

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

speak("VisionAid AI started. Point camera to the scene.")

last_scan_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("VisionAid AI", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # =========================
    # AUTO SCAN MODE
    # =========================
    current_time = time.time()

    if current_time - last_scan_time >= SCAN_INTERVAL:

        last_scan_time = current_time

        # ===== YOLO PERSON DETECTION =====
        results = model(frame)
        people_count = 0

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] == "person":
                people_count += 1

        # ===== OCR TEXT READING =====
        text_result = ""

        if not LITE_MODE:
            ocr_results = reader.readtext(frame)
            text_result = " ".join([t[1] for t in ocr_results]) if ocr_results else ""

        # =========================
        # RESPONSE (SHORT & CLEAR)
        # =========================
        if people_count > 0:
            speak(f"{people_count} person detected.")
        else:
            speak("No people detected.")

        if text_result:
            speak(f"Text detected: {text_result}")

        if LITE_MODE:
            speak("Lite mode active.")

cap.release()
cv2.destroyAllWindows()