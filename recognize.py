import numpy as np
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier

X = []
y = []

for file in os.listdir("signs"):
    if file.endswith(".npy"):
        data = np.load(f"signs/{file}")
        label = file.replace("sign_", "").replace(".npy", "")

        for sample in data:
            X.append(sample)
            y.append(label)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

with open("model/gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained successfully!")
