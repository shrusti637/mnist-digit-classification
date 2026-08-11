# MNIST Digit Classification

A handwritten digit classification project built with Python and machine learning. The project trains an MLP classifier on the MNIST dataset and provides an interactive Streamlit application for digit prediction and model analysis.

## 📌 Project Overview

This project classifies handwritten digits from 0 to 9 using the MNIST dataset.

The training pipeline uses `MLPClassifier` from Scikit-learn with data augmentation. A Streamlit web application allows users to draw a digit and receive a prediction along with prediction probabilities and model analysis.

## 🛠️ Technologies Used

- Python
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- Pillow
- Joblib
- Streamlit
- Jupyter Notebook

## ✨ Features

- Handwritten digit classification
- Interactive digit drawing interface
- Prediction probability visualization
- Confusion matrix analysis
- Model metrics dashboard
- Sample prediction visualization
- Data augmentation during training
- Interactive Streamlit interface

## 🧠 Machine Learning Model

The project uses a Multi-Layer Perceptron (MLP) classifier from Scikit-learn.

The training process includes:

1. Loading the MNIST dataset
2. Preprocessing and normalization
3. Data augmentation
4. Training the MLP classifier
5. Evaluating the model
6. Saving model results and supporting data
7. Using the trained model for predictions through Streamlit

## 📁 Project Structure

```text
mnist-digit-classification/
│
├── app.py
├── train.py
├── mnist_training.ipynb
├── mnist_training.pdf
├── requirements.txt
├── confusion_matrix.npy
├── metrics.json
├── samples.json
├── .gitignore
└── README.md# mnist-digit-classification
Handwritten digit classification 
