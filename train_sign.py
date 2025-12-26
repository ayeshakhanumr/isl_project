import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# USER INPUT
# -----------------------------
sign_name = input("Enter sign name (example: hello, thankyou): ").strip().lower()

os.makedirs("signs", exist_ok=True)
save_path = f"signs/sign_{sign_name}.npy"

existing_data = None

if os.path.exists(save_path):
    print("⚠ This sign already exists.")
    print("1️⃣ Retrain from scratch")
    print("2️⃣ Add more samples to existing data")
    choice = input("Choose option (1 or 2): ").strip()

    if choice == "2":
        existing_data = np.load(save_path)
        print(f"📊 Existing samples: {len(existing_data)}")
    elif choice == "1":
        print("🔁 Retraining from scratch...")
    else:
        print("❌ Invalid choice")
        exit()

# -----------------------------
# TRAINING
# -----------------------------
data = []

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    cap = cv2.VideoCapture(0)
    print("🎥 Camera started")
    print("👉 Press S to save sample | Q to quit")

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
                mp_draw.draw_landmarks(
                    frame, handLms, mp_hands.HAND_CONNECTIONS
                )
                for lm in handLms.landmark:
                    features.extend([lm.x, lm.y, lm.z])

        # Normalize feature length
        if len(features) == 63:
            features.extend([0.0] * 63)   # pad for one hand

        cv2.imshow("Training Window", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and len(features) == 126:
            data.append(features)
            print(f"✅ Sample saved: {len(data)}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# SAVE DATA
# -----------------------------
if len(data) == 0:
    print("❌ No samples collected")
    exit()

new_data = np.array(data)

if existing_data is not None:
    final_data = np.vstack([existing_data, new_data])
    print(f"📈 Total samples after merge: {len(final_data)}")
else:
    final_data = new_data
    print("🆕 Training from scratch")

np.save(save_path, final_data)

print("🎉 Training completed successfully!")
print(f"📁 Saved at: {save_path}")
print(f"📐 Feature size: {final_data.shape[1]}")
