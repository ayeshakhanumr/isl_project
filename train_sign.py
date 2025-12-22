import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
draw = mp.solutions.drawing_utils

sign_name = input("Enter sign name (example: hello, thumbs_up): ")
save_path = f"signs/sign_{sign_name}.npy"

data = []

cap = cv2.VideoCapture(0)
print("Press S to save sample | Q to quit")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        landmarks = []

        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Training Mode", frame)

    key = cv2.waitKey(1)
    if key == ord('s') and result.multi_hand_landmarks:
        data.append(landmarks)
        print(f"Sample saved: {len(data)}")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

np.save(save_path, np.array(data))
print(f"Dataset saved as {save_path}")
