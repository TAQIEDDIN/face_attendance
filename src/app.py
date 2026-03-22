"""
app.py — ALL-IN-ONE
  • Camera capture  (SharedCamera thread)
  • Face recognition (RecognitionThread — runs InsightFace on shared frames)
  • Flask web server  (dashboard + API)

Run with a SINGLE terminal:
    python src/app.py

recognize.py is NO LONGER NEEDED.
"""

import os
import cv2
import time
import threading
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, Response, jsonify, request
from database import get_connection
from attendance import log_event, save_unknown_image

# ----------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------
BASE_DIR       = os.path.dirname(os.path.dirname(__file__))
TEMPLATES_DIR  = os.path.join(BASE_DIR, "templates")
STATIC_DIR     = os.path.join(BASE_DIR, "static")
EMPLOYEES_DIR  = os.path.join(BASE_DIR, "data", "employees")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "embeddings")
CHECK_IN_DIR   = os.path.join(BASE_DIR, "data", "events", "check_in")
CHECK_OUT_DIR  = os.path.join(BASE_DIR, "data", "events", "check_out")
UNKNOWN_DIR    = os.path.join(BASE_DIR, "data", "unknown")
HEARTBEAT_FILE = os.path.join(BASE_DIR, "data", "recognizer.heartbeat")

# Support old src/ path for embeddings
EMB_PATH   = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
NAMES_PATH = os.path.join(EMBEDDINGS_DIR, "names.npy")
if not os.path.exists(EMB_PATH):
    _old = os.path.join(BASE_DIR, "src", "embeddings.npy")
    if os.path.exists(_old):
        EMB_PATH   = _old
        NAMES_PATH = os.path.join(BASE_DIR, "src", "names.npy")

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(EMPLOYEES_DIR,  exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024


# ================================================================
# 1. SHARED CAMERA  (single VideoCapture — DirectShow backend)
# ================================================================
class SharedCamera:
    def __init__(self, src=0):
        # Try DirectShow first (Windows), fall back to default
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 25)

        self.lock    = threading.Lock()
        self._frame  = None
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self._frame = frame
            else:
                time.sleep(0.03)

    def get_frame(self):
        with self.lock:
            return self._frame.copy() if self._frame is not None else None

    def is_open(self):
        return self.cap.isOpened() and self._frame is not None

    def release(self):
        self.running = False
        self.cap.release()


camera = SharedCamera(0)


# ================================================================
# 2. FACE RECOGNITION THREAD  (reads from SharedCamera — no extra cap)
# ================================================================
class RecognitionThread(threading.Thread):
    THRESHOLD       = 0.45
    COOLDOWN        = 8     # seconds between events per person
    UNKNOWN_COOLDOWN= 5     # seconds between unknown saves

    def __init__(self):
        super().__init__(daemon=True)
        self.running              = True
        self.fa                   = None          # loaded lazily
        self.known_embeddings     = None
        self.known_names          = None
        self.person_states        = {}
        self.last_unknown_time    = None
        self._load_embeddings()

    # ── Embedding helpers ──────────────────────────────────────
    def _load_embeddings(self):
        if os.path.exists(EMB_PATH) and os.path.exists(NAMES_PATH):
            self.known_embeddings = np.array(
                np.load(EMB_PATH, allow_pickle=True), dtype=np.float32)
            self.known_names = np.load(NAMES_PATH, allow_pickle=True)
            print(f"[Recognition] Loaded {len(self.known_names)} enrolled person(s): "
                  f"{list(self.known_names)}")
        else:
            self.known_embeddings = np.array([], dtype=np.float32)
            self.known_names      = np.array([])
            print("[Recognition] No embeddings found — enroll employees first.")

    def reload_embeddings(self):
        """Called by /api/enroll after new enrollment."""
        self._load_embeddings()

    # ── InsightFace (lazy load so Flask starts fast) ────────────
    def _ensure_fa(self):
        if self.fa is None:
            from insightface.app import FaceAnalysis
            self.fa = FaceAnalysis(name="buffalo_l")
            self.fa.prepare(ctx_id=-1, det_size=(640, 640))
            print("[Recognition] InsightFace ready.")

    # ── Cosine similarity ───────────────────────────────────────
    @staticmethod
    def cosine_sim(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return -1.0
        return float(np.dot(a / na, b / nb))

    # ── Heartbeat ───────────────────────────────────────────────
    def _heartbeat(self):
        try:
            with open(HEARTBEAT_FILE, 'w') as f:
                f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        except OSError:
            pass

    # ── Main loop ───────────────────────────────────────────────
    def run(self):
        self._ensure_fa()
        hb_counter = 0
        print("[Recognition] Thread started.")

        while self.running:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            # Heartbeat every ~25 frames
            hb_counter += 1
            if hb_counter >= 25:
                self._heartbeat()
                hb_counter = 0

            # Skip if no enrolled people
            if len(self.known_names) == 0:
                time.sleep(0.5)
                continue

            now   = datetime.now()
            faces = self.fa.get(frame)

            for face in faces:
                box       = face.bbox.astype(int)
                x1,y1,x2,y2 = box
                emb = face.embedding.astype(np.float32)

                # Match against known
                best_score, best_name = -1.0, "Unknown"
                for i, known_emb in enumerate(self.known_embeddings):
                    s = self.cosine_sim(emb, known_emb)
                    if s > best_score:
                        best_score = s
                        best_name  = self.known_names[i]

                if best_score < self.THRESHOLD:
                    best_name = "Unknown"

                if best_name != "Unknown":
                    if best_name not in self.person_states:
                        log_event(best_name, "check-in", frame)
                        self.person_states[best_name] = {
                            "last_event": "check-in", "last_seen_time": now}
                        print(f"[CHECK-IN]  {best_name}  ({best_score:.2f})")
                    else:
                        diff = (now - self.person_states[best_name]["last_seen_time"]).total_seconds()
                        if diff > self.COOLDOWN:
                            prev = self.person_states[best_name]["last_event"]
                            ev   = "check-out" if prev == "check-in" else "check-in"
                            log_event(best_name, ev, frame)
                            self.person_states[best_name].update(
                                {"last_event": ev, "last_seen_time": now})
                            print(f"[{ev.upper().replace('-','_')}] {best_name}  ({best_score:.2f})")
                else:
                    save_unk = (
                        self.last_unknown_time is None or
                        (now - self.last_unknown_time).total_seconds() > self.UNKNOWN_COOLDOWN
                    )
                    if save_unk:
                        crop = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                        if crop.size > 0:
                            save_unknown_image(crop)
                            self.last_unknown_time = now

            time.sleep(1 / 15)   # ~15 fps recognition


# Start recognition thread immediately
recognizer = RecognitionThread()
recognizer.start()


# ================================================================
# 3. FLASK — VIDEO FEED
# ================================================================
def generate_frames():
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.04)
            continue
        h = frame.shape[0]
        cv2.putText(frame, "ENTRANCE",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        cv2.putText(frame, time.strftime("%H:%M:%S"),
                    (frame.shape[1] - 80, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(1 / 25)


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ================================================================
# 4. ENROLL API
# ================================================================
@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "error": "Name required"}), 400

    files = request.files.getlist("images")
    if not files or all(f.filename == '' for f in files):
        return jsonify({"ok": False, "error": "At least one image required"}), 400

    person_dir = os.path.join(EMPLOYEES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    saved = []
    for i, f in enumerate(files):
        if not f.filename:
            continue
        ext = os.path.splitext(f.filename)[1].lower() or '.jpg'
        if ext not in ('.jpg','.jpeg','.png','.bmp','.webp'):
            continue
        path = os.path.join(person_dir,
                            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:02d}{ext}")
        f.save(path)
        saved.append(path)

    if not saved:
        return jsonify({"ok": False, "error": "No valid images"}), 400

    # Use recognizer's InsightFace instance (already loaded)
    recognizer._ensure_fa()
    fa = recognizer.fa

    embeddings, face_count = [], 0
    for path in saved:
        img   = cv2.imread(path)
        if img is None:
            continue
        faces = fa.get(img)
        if not faces:
            continue
        largest = max(faces, key=lambda fc: (fc.bbox[2]-fc.bbox[0])*(fc.bbox[3]-fc.bbox[1]))
        embeddings.append(largest.embedding.astype(np.float32))
        face_count += 1

    if not embeddings:
        return jsonify({"ok": False,
                        "error": f"No face detected in {len(saved)} image(s). "
                                 "Use clear front-facing photos."}), 400

    avg = np.mean(embeddings, axis=0)
    avg = avg / np.linalg.norm(avg)

    # Load + update .npy
    if os.path.exists(EMB_PATH) and os.path.exists(NAMES_PATH):
        all_embs  = list(np.load(EMB_PATH,   allow_pickle=True))
        all_names = list(np.load(NAMES_PATH, allow_pickle=True))
    else:
        all_embs, all_names = [], []

    if name in all_names:
        all_embs[all_names.index(name)] = avg
        action = "updated"
    else:
        all_embs.append(avg)
        all_names.append(name)
        action = "enrolled"

    np.save(EMB_PATH,   np.array(all_embs,  dtype=np.float32))
    np.save(NAMES_PATH, np.array(all_names))

    # Hot-reload embeddings in recognition thread (no restart needed)
    recognizer.reload_embeddings()

    return jsonify({"ok": True, "action": action, "name": name,
                    "faces_used": face_count, "images_sent": len(saved),
                    "total_staff": len(all_names)})


# ================================================================
# 5. STATUS API
# ================================================================
def _count_images(folder, minutes=60):
    if not os.path.isdir(folder):
        return 0
    cutoff = (datetime.now() - timedelta(minutes=minutes)).timestamp()
    return sum(1 for f in os.listdir(folder)
               if os.path.getmtime(os.path.join(folder, f)) >= cutoff)


def _last_ago(folder):
    if not os.path.isdir(folder):
        return None
    times = [os.path.getmtime(os.path.join(folder, f)) for f in os.listdir(folder)]
    return round(time.time() - max(times)) if times else None


def _db_check():
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM attendance_events WHERE timestamp LIKE ?",
                    (datetime.now().strftime("%Y-%m-%d") + "%",))
        c = cur.fetchone()[0]
        conn.close()
        return True, c
    except Exception:
        return False, 0


@app.route("/api/status")
def api_status():
    cam_ok    = camera.is_open()
    # Recognizer is always "alive" — it's a thread in this process
    rec_alive = recognizer.is_alive()
    db_ok, db_today = _db_check()
    return jsonify({
        "camera":     {"ok": cam_ok,    "label": "CONNECTED" if cam_ok    else "DISCONNECTED"},
        "recognizer": {"ok": rec_alive, "label": "RUNNING"   if rec_alive else "STOPPED",
                       "seconds_ago": 0 if rec_alive else None},
        "database":   {"ok": db_ok,     "label": "ONLINE"    if db_ok     else "ERROR",
                       "events_today": db_today},
        "images": {
            "checkin":  {"count": _count_images(CHECK_IN_DIR,  60), "last_sec": _last_ago(CHECK_IN_DIR)},
            "checkout": {"count": _count_images(CHECK_OUT_DIR, 60), "last_sec": _last_ago(CHECK_OUT_DIR)},
            "unknown":  {"count": _count_images(UNKNOWN_DIR,   60), "last_sec": _last_ago(UNKNOWN_DIR)},
        },
        "ts": datetime.now().strftime("%H:%M:%S")
    })


# ================================================================
# 6. PAGE
# ================================================================
def _enrolled_count():
    if not os.path.isdir(EMPLOYEES_DIR):
        return 0
    return sum(1 for d in os.listdir(EMPLOYEES_DIR)
               if os.path.isdir(os.path.join(EMPLOYEES_DIR, d)))


def _stats(cursor, status_dict):
    cursor.execute("SELECT COUNT(*) FROM attendance_events WHERE timestamp LIKE ?",
                   (datetime.now().strftime("%Y-%m-%d") + "%",))
    return {
        "total_enrolled": _enrolled_count(),
        "currently_in":   sum(1 for e in status_dict.values() if e == "check-in"),
        "currently_out":  sum(1 for e in status_dict.values() if e == "check-out"),
        "events_today":   cursor.fetchone()[0],
        "unknown_today":  _count_images(UNKNOWN_DIR, 1440),
    }


def _hourly(cursor):
    now = datetime.now()
    labels, h_in, h_out = [], [], []
    for i in range(11, -1, -1):
        dt  = now - timedelta(hours=i)
        pat = dt.strftime("%Y-%m-%d %H:%%")
        cursor.execute("SELECT COUNT(*) FROM attendance_events WHERE timestamp LIKE ? AND event='check-in'", (pat,))
        h_in.append(cursor.fetchone()[0])
        cursor.execute("SELECT COUNT(*) FROM attendance_events WHERE timestamp LIKE ? AND event='check-out'", (pat,))
        h_out.append(cursor.fetchone()[0])
        labels.append(dt.strftime("%H:00"))
    return labels, h_in, h_out


@app.route("/")
def index():
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name, timestamp, event FROM attendance_events ORDER BY id DESC LIMIT 50")
    rows = cursor.fetchall()
    cursor.execute("""
        SELECT name, event FROM attendance_events
        WHERE id IN (SELECT MAX(id) FROM attendance_events GROUP BY name)
    """)
    status_dict = {n: e for n, e in cursor.fetchall()}
    stats       = _stats(cursor, status_dict)
    h_lbl, h_in, h_out = _hourly(cursor)
    conn.close()
    return render_template("index.html",
                           data=rows, status=status_dict, stats=stats,
                           hourly_labels=h_lbl, hourly_in=h_in, hourly_out=h_out,
                           enumerate=enumerate)


if __name__ == "__main__":
    print("=" * 50)
    print("  Face Attendance — ALL-IN-ONE")
    print("  Dashboard: http://127.0.0.1:5000")
    print("  recognize.py is NO LONGER needed")
    print("=" * 50)
    app.run(debug=False, threaded=True, port=5000)