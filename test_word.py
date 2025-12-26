import cv2
import mediapipe as mp
import numpy as np
import os

# -----------------------------
# CONFIG
# -----------------------------
SIGN_DIR = "signs"
FEATURE_SIZE = 126   # always use 2-hand size

# -----------------------------
# Load trained sign data safely
# -----------------------------
X = []
y = []

for file in os.listdir(SIGN_DIR):
    if not file.endswith(".npy"):
        continue

    label = file.replace("sign_", "").replace(".npy", "")
    data = np.load(os.path.join(SIGN_DIR, file))

    # ensure 2D
    if len(data.shape) != 2:
        continue

    for sample in data:
        if len(sample) == 63:
            # pad 1-hand samples
            sample = np.concatenate([sample, np.zeros(63)])
        elif len(sample) != FEATURE_SIZE:
            continue

        X.append(sample)
        y.append(label)

X = np.array(X)
y = np.array(y)

print(f"✅ Loaded {len(X)} samples from {len(set(y))} signs")

# -----------------------------
# Prediction (distance-based)
# -----------------------------
def predict_word(sample):
    dists = np.linalg.norm(X - sample, axis=1)
    idx = np.argmin(dists)

    label = y[idx]
    confidence = max(0, 100 - int(dists[idx] * 100))

    return label, confidence

# -----------------------------
# MediaPipe setup
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        features = []

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                for lm in handLms.landmark:
                    features.extend([lm.x, lm.y, lm.z])

            # normalize to 126
            if len(features) == 63:
                features.extend([0.0] * 63)

            if len(features) == FEATURE_SIZE:
                word, acc = predict_word(np.array(features))

                cv2.putText(frame, f"Word: {word}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

                cv2.putText(frame, f"Accuracy: {acc}%", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 255, 0), 2)

        cv2.putText(frame, "Press Q to Quit",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        cv2.imshow("ISL Word Test", frame)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

cap.release()
cv2.destroyAllWindows()
