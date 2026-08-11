import torch
from torchvision import datasets, transforms
import os

print("Checking data directory...")
if not os.path.exists('./data'):
    os.makedirs('./data')

print("Attempting to load MNIST...")
try:
    datasets.MNIST('./data', train=True, download=True)
    print("Successfully loaded/downloaded MNIST.")
except Exception as e:
    print(f"Error: {e}")
