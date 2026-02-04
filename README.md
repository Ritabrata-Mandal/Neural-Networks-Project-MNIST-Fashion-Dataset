# Neural Networks Project – Fashion MNIST Dataset

This project implements a **feed-forward neural network** using **Keras (TensorFlow backend)** to classify images from the **Fashion-MNIST dataset**.  
It is designed as a **learning-oriented project** to understand neural network fundamentals such as layers, parameters, activations, and model evaluation.

---

## 📌 Dataset
**Fashion-MNIST** is a dataset of grayscale images representing clothing items.

- Image size: **28 × 28**
- Classes: **10**
- Training samples: **60,000**
- Test samples: **10,000**

Example classes include:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

---

## 🧠 Model Architecture
The neural network follows a **Sequential** architecture:

- **Input Layer:** 28×28 images
- **Flatten Layer:** Converts images to a 1D vector (784 features)
- **Dense Layer:** 300 neurons, ReLU activation
- **Dense Layer:** 300 neurons, ReLU activation
- **Output Layer:** 10 neurons, Softmax activation

This architecture is suitable for understanding **fully connected neural networks**.

---

## ⚙️ Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Jupyter Notebook
- Anaconda

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Ritabrata-Mandal/Neural-Networks-Project-MNIST-Fashion-Dataset.git
