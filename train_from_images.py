import cv2
import mediapipe as mp
import numpy as np
import os
from tkinter import Tk, filedialog

mp_hands = mp.solutions.hands

def select_images():
    Tk().withdraw()
    return filedialog.askopenfilenames(
        title="Select training images",
        filetypes=[("Images", "*.jpg *.jpeg *.png")]
    )

sign_name = input("Enter sign name: ").strip().lower()
image_paths = select_images()

if not image_paths:
    print("❌ No images selected")
    exit()

features = []

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.3
) as hands:

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (640, 480))

        # 🔥 Contrast enhancement
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(2.0, (8, 8))
        gray = clahe.apply(gray)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        result = hands.process(rgb)

        if not result.multi_hand_landmarks:
            print(f"⚠ No hand detected in {os.path.basename(img_path)}")
            continue

        sample = []

        for handLms in result.multi_hand_landmarks:
            for lm in handLms.landmark:
                sample.extend([lm.x, lm.y, lm.z])

        # 🔒 FORCE 2-HAND FEATURE SIZE
        if len(sample) == 63:
            sample.extend([0.0] * 63)

        if len(sample) == 126:
            features.append(sample)

print(f"✅ Valid samples: {len(features)}")

if len(features) < 10:
    print("❌ Not enough good samples. Capture clearer images.")
    exit()

os.makedirs("signs", exist_ok=True)
save_path = f"signs/sign_{sign_name}.npy"
np.save(save_path, np.array(features))

print(f"🎉 Saved: {save_path}")
print(f"📐 Feature size: {len(features[0])}")
