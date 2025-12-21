import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Load all sign datasets
X = []  # landmarks
y = []  # labels

dataset_path = "signs"
for file in os.listdir(dataset_path):
    if file.endswith(".npy"):
        label = file.replace("sign_", "").replace(".npy", "")
        data = np.load(os.path.join(dataset_path, file))
        X.append(data)
        y += [label]*len(data)

if len(X) == 0:
    print("No datasets found! Exiting...")
    exit()

X = np.vstack(X)
y = np.array(y)

# Train KNN classifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X, y)

# Start webcam
cap = cv2.VideoCapture(0)
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
            pred = clf.predict([landmark_array])[0]  # predict sign
            cv2.putText(frame, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Hand Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    cv2.imshow("ISL Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
