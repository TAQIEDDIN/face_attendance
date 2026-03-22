import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "employees")

EMB_DIR = os.path.join(BASE_DIR, "data", "embeddings")
EMB_PATH = os.path.join(EMB_DIR, "embeddings.npy")
NAMES_PATH = os.path.join(EMB_DIR, "names.npy")

# تأكد folder موجود
os.makedirs(EMB_DIR, exist_ok=True)

# -----------------------------
# InsightFace init
# -----------------------------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(640, 640))

embeddings = []
names = []

if not os.path.exists(DATA_PATH):
    print(f"Error: Employees folder not found -> {DATA_PATH}")
    exit()

for person_name in os.listdir(DATA_PATH):
    person_path = os.path.join(DATA_PATH, person_name)

    if not os.path.isdir(person_path):
        continue

    print(f"[INFO] Processing: {person_name}")

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Could not read image: {img_path}")
            continue

        faces = app.get(img)

        if len(faces) == 0:
            print(f"[WARNING] No face found in: {img_path}")
            continue

        embedding = faces[0].embedding.astype(np.float32)
        embeddings.append(embedding)
        names.append(person_name)

# حفظ
np.save(EMB_PATH, np.array(embeddings, dtype=np.float32))
np.save(NAMES_PATH, np.array(names))

print("Enrollment done ✅")
print(f"Saved embeddings to: {EMB_PATH}")
print(f"Saved names to: {NAMES_PATH}")
print(f"Total embeddings: {len(embeddings)}")