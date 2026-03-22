import os
import cv2
from datetime import datetime
from database import get_connection, init_db

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

CHECK_IN_DIR = os.path.join(BASE_DIR, "data", "events", "check_in")
CHECK_OUT_DIR = os.path.join(BASE_DIR, "data", "events", "check_out")
UNKNOWN_DIR = os.path.join(BASE_DIR, "data", "unknown")

os.makedirs(CHECK_IN_DIR, exist_ok=True)
os.makedirs(CHECK_OUT_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

init_db()


def save_event_image(frame, name, event_type):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    if event_type == "check-in":
        folder = CHECK_IN_DIR
    elif event_type == "check-out":
        folder = CHECK_OUT_DIR
    else:
        return

    filename = f"{name}_{timestamp}.jpg"
    path = os.path.join(folder, filename)
    cv2.imwrite(path, frame)


def save_unknown_image(frame):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"unknown_{timestamp}.jpg"
    path = os.path.join(UNKNOWN_DIR, filename)
    cv2.imwrite(path, frame)


def log_event(name, event_type, frame=None):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO attendance_events (name, timestamp, event)
        VALUES (?, ?, ?)
    """, (name, timestamp, event_type))

    conn.commit()
    conn.close()

    print(f"[{event_type.upper()}] {name} | {timestamp}")

    if frame is not None:
        save_event_image(frame, name, event_type)