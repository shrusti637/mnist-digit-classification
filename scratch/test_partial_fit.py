import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
import os

# Load the current model
model_path = "c:\\Users\\Shrusti\\Documents\\mnist_handwritten_system\\mnist_model.joblib"
if not os.path.exists(model_path):
    print("Model not found")
    exit()

mlp = joblib.load(model_path)

# Create a random 'wrong' sample (simulating the user's '4' that predicts '6')
# For demonstration, we'll just take a random sample
np.random.seed(42)
sample = np.random.rand(1, 784)
original_pred = mlp.predict(sample)[0]
original_probs = mlp.predict_proba(sample)[0]

print(f"Original Prediction: {original_pred}")
print(f"Probabilities: {original_probs[int(original_pred)]: .4f}")

# Target label (e.g., if it predicted 6, we want to teach it 4)
target_label = '4' if original_pred != '4' else '2'
target_idx = int(target_label)

print(f"Teaching digit: {target_label}")

# Try 1 iteration
mlp.partial_fit(sample, [target_label])
new_pred = mlp.predict(sample)[0]
new_probs = mlp.predict_proba(sample)[0]
print(f"After 1 iteration: Prediction={new_pred}, Target Prob={new_probs[target_idx]: .4f}")

# Try more iterations
for i in range(2, 21):
    mlp.partial_fit(sample, [target_label])
    if i % 5 == 0:
        p = mlp.predict(sample)[0]
        pr = mlp.predict_proba(sample)[0]
        print(f"After {i} iterations: Prediction={p}, Target Prob={pr[target_idx]: .4f}")
