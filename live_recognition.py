import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque

# ---------------------------
# MediaPipe setup
# ---------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

SIGNS_DIR = "signs"

# ---------------------------
# Load datasets safely
# ---------------------------
datasets = {}   # {feature_size: {"X": [], "y": []}}

for file in os.listdir(SIGNS_DIR):
    if not file.endswith(".npy"):
        continue

    label = file.replace("sign_", "").replace(".npy", "")
    path = os.path.join(SIGNS_DIR, file)

    data = np.load(path, allow_pickle=True)

    # ❌ skip empty files
    if data.size == 0:
        print(f"⚠ Skipping empty file: {file}")
        continue

    # fix 1D shape
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # must be 2D
    if data.ndim != 2:
        print(f"⚠ Skipping invalid shape: {file} -> {data.shape}")
        continue

    feat_size = data.shape[1]

    if feat_size not in datasets:
        datasets[feat_size] = {"X": [], "y": []}

    datasets[feat_size]["X"].append(data)
    datasets[feat_size]["y"].extend([label] * len(data))

# ---------------------------
# Build final arrays
# ---------------------------
models = {}  # feature_size → (X, y)

for feat_size, d in datasets.items():
    X = np.vstack(d["X"])
    y = np.array(d["y"])
    models[feat_size] = (X, y)
    print(f"✅ Loaded model for {feat_size} features → {len(y)} samples")

if not models:
    print("❌ No valid datasets found!")
    exit()

# ---------------------------
# Simple KNN
# ---------------------------
def predict_knn(X, y, sample):
    dists = np.linalg.norm(X - sample, axis=1)
    return y[np.argmin(dists)]

# ---------------------------
# Sentence buffer
# ---------------------------
sentence = deque(maxlen=3)
last_pred = None

# ---------------------------
# Webcam
# ---------------------------
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        features = []

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                for lm in handLms.landmark:
                    features.extend([lm.x, lm.y, lm.z])

            features = np.array(features)

            # match correct model
            if features.size in models:
                X, y = models[features.size]
                pred = predict_knn(X, y, features)

                if pred != last_pred:
                    sentence.append(pred)
                    last_pred = pred

        cv2.putText(
            frame,
            "Sentence: " + " ".join(sentence),
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Live ISL Recognition", frame)

        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break

cap.release()
cv2.destroyAllWindows()
