import cv2
import mediapipe as mp
import numpy as np
import os

# Path to your dataset
SIGN_NAME = "hello"
DATA_PATH = f"signs/sign_{SIGN_NAME}.npy"

# Load trained data
if os.path.exists(DATA_PATH):
    data = np.load(DATA_PATH)
else:
    print(f"Dataset {DATA_PATH} not found!")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

print(f"Testing sign: {SIGN_NAME}")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Convert landmarks to array
            landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            # Compare with saved dataset (simple nearest neighbor)
            distances = np.linalg.norm(data - landmark_array, axis=1)
            min_dist = np.min(distances)
            if min_dist < 0.9:  # threshold, adjust if needed
                cv2.putText(frame, SIGN_NAME, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)

    cv2.imshow("Test Sign", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
