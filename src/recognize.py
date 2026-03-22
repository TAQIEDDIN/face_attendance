"""
recognize.py — reads frames from app.py's /frame endpoint.

WHY: Windows (MSMF) does NOT allow two processes to open the same
camera at once. app.py already owns VideoCapture(0) and serves each
latest frame at  GET http://127.0.0.1:5000/frame  as a JPEG.
This script just fetches that JPEG, decodes it, and runs InsightFace.

Run order:
  Terminal 1 →  python src/app.py        (starts camera + web server)
  Terminal 2 →  python src/recognize.py  (reads frames from app.py)
"""

import os
import cv2
import time
import numpy as np
import urllib.request
from datetime import datetime
from insightface.app import FaceAnalysis
from attendance import log_event, save_unknown_image

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
BASE_DIR       = os.path.dirname(os.path.dirname(__file__))
EMB_PATH       = os.path.join(BASE_DIR, "data", "embeddings", "embeddings.npy")
NAMES_PATH     = os.path.join(BASE_DIR, "data","embeddings", "names.npy")
HEARTBEAT_FILE = os.path.join(BASE_DIR, "data", "recognizer.heartbeat")

# URL of app.py's raw frame endpoint
FRAME_URL = "http://127.0.0.1:5000/frame"

# ----------------------------------------------------------------
# Check enrolled data
# ----------------------------------------------------------------
if not os.path.exists(EMB_PATH) or not os.path.exists(NAMES_PATH):
    print("[ERROR] embeddings.npy or names.npy not found. Run enroll.py first.")
    exit(1)

known_embeddings = np.array(np.load(EMB_PATH,   allow_pickle=True), dtype=np.float32)
known_names      = np.load(NAMES_PATH, allow_pickle=True)

# ----------------------------------------------------------------
# InsightFace
# ----------------------------------------------------------------
fa = FaceAnalysis(name="buffalo_l")
fa.prepare(ctx_id=-1, det_size=(640, 640))
print("[INFO] InsightFace ready.")


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
def write_heartbeat():
    try:
        with open(HEARTBEAT_FILE, 'w') as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    except OSError:
        pass


def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return -1.0
    return float(np.dot(a / na, b / nb))


def fetch_frame(retries=5):
    """
    Fetch latest JPEG from app.py and decode to numpy BGR array.
    Returns None if app.py is not yet ready.
    """
    for _ in range(retries):
        try:
            with urllib.request.urlopen(FRAME_URL, timeout=2) as resp:
                jpg = np.frombuffer(resp.read(), dtype=np.uint8)
                frame = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
        except Exception:
            pass
        time.sleep(0.5)
    return None


# ----------------------------------------------------------------
# Wait for app.py to start
# ----------------------------------------------------------------
print("[INFO] Waiting for app.py to start (GET /frame)...")
while True:
    f = fetch_frame(retries=1)
    if f is not None:
        print("[INFO] app.py is ready. Starting recognition loop.")
        break
    print("[INFO] app.py not ready yet, retrying in 2s...")
    time.sleep(2)


# ----------------------------------------------------------------
# Recognition loop
# ----------------------------------------------------------------
THRESHOLD         = 0.45
COOLDOWN          = 8      # seconds between same-person events
UNKNOWN_COOLDOWN  = 5      # seconds between unknown saves

person_states          = {}
last_unknown_save_time = None
heartbeat_counter      = 0
HEARTBEAT_EVERY        = 25  # frames

print("[INFO] Recognition running. Press Ctrl-C to stop.")

while True:
    frame = fetch_frame(retries=3)

    if frame is None:
        print("[WARN] Could not get frame from app.py. Retrying...")
        time.sleep(0.5)
        continue

    # ── Heartbeat ──
    heartbeat_counter += 1
    if heartbeat_counter >= HEARTBEAT_EVERY:
        write_heartbeat()
        heartbeat_counter = 0

    now   = datetime.now()
    faces = fa.get(frame)

    for face in faces:
        box       = face.bbox.astype(int)
        x1,y1,x2,y2 = box
        emb       = face.embedding.astype(np.float32)

        # Find best match
        best_score = -1.0
        best_name  = "Unknown"
        for i, known_emb in enumerate(known_embeddings):
            s = cosine_similarity(emb, known_emb)
            if s > best_score:
                best_score = s
                best_name  = known_names[i]

        if best_score < THRESHOLD:
            best_name = "Unknown"

        # ── Known person ──
        if best_name != "Unknown":
            if best_name not in person_states:
                log_event(best_name, "check-in", frame)
                person_states[best_name] = {
                    "last_event":    "check-in",
                    "last_seen_time": now
                }
                print(f"[CHECK-IN]  {best_name}  score={best_score:.2f}")
            else:
                diff = (now - person_states[best_name]["last_seen_time"]).total_seconds()
                if diff > COOLDOWN:
                    prev = person_states[best_name]["last_event"]
                    new_event = "check-out" if prev == "check-in" else "check-in"
                    log_event(best_name, new_event, frame)
                    person_states[best_name]["last_event"]    = new_event
                    person_states[best_name]["last_seen_time"] = now
                    print(f"[{new_event.upper().replace('-','_')}] {best_name}  score={best_score:.2f}")

        # ── Unknown person ──
        else:
            save_unknown = (
                last_unknown_save_time is None or
                (now - last_unknown_save_time).total_seconds() > UNKNOWN_COOLDOWN
            )
            if save_unknown:
                face_crop = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                if face_crop.size > 0:
                    save_unknown_image(face_crop)
                    last_unknown_save_time = now
                    print(f"[UNKNOWN]   score={best_score:.2f}  image saved")

    # Throttle to ~15 fps (camera already capped at 25 in app.py)
    time.sleep(1 / 15)