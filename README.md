# 🧠 ANN Reuters Classifier

A neural network classifier for the Reuters news dataset using Keras.

## 📝 Description

This repository contains a neural network implementation for classifying Reuters news articles into 46 different categories. It uses a simple yet effective architecture with two dense hidden layers to achieve multi-class classification of news content.

The classifier demonstrates:
- Text data preprocessing and vectorization
- Multi-class classification with neural networks
- Training visualization and evaluation

## ✨ Features

- Automatic loading and preprocessing of the Reuters dataset
- Text vectorization using one-hot encoding
- Neural network model with configurable architecture
- Training and validation metrics visualization
- Multi-class classification capability

## 🔧 Prerequisites

- Python 3.6+
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib

## 🚀 Setup Guide

1. Clone this repository:
   ```bash
   git clone https://github.com/corticalstack/ANNReutersClassifier.git
   cd ANNReutersClassifier
   ```

2. Install the required dependencies:
   ```bash
   pip install tensorflow numpy matplotlib
   ```

3. Run the classifier:
   ```bash
   python main.py
   ```

## 🏗️ Architecture

The neural network architecture consists of:
- Input layer accepting 10,000-dimensional vectors (bag-of-words representation)
- First hidden layer with 64 neurons and ReLU activation
- Second hidden layer with 64 neurons and ReLU activation
- Output layer with 46 neurons (one per category) and softmax activation

The model is compiled with:
- RMSprop optimizer
- Categorical crossentropy loss function
- Accuracy metric

## 📊 How It Works

1. **Data Loading**: The Reuters dataset is loaded with a vocabulary limited to the 10,000 most frequent words.
2. **Text Preprocessing**: Each news article is converted into a fixed-length vector using one-hot encoding.
3. **Label Encoding**: The category labels are one-hot encoded.
4. **Model Training**: The neural network is trained for 20 epochs with a validation split.
5. **Visualization**: Training and validation loss/accuracy are plotted to evaluate model performance.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Resources

- [Keras Documentation](https://keras.io/)
- [Reuters-21578 Dataset Information](https://keras.io/api/datasets/reuters/)
- [Neural Network Introduction](https://www.tensorflow.org/tutorials/keras/classification)
