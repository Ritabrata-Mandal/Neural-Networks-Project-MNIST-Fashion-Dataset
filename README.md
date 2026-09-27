# Fashion MNIST Classification using CNN + ANN

A deep learning project for classifying images from the Fashion-MNIST
dataset using a Convolutional Neural Network (CNN) followed by an
Artificial Neural Network (ANN) classification head.

## Project Overview

Fashion-MNIST contains 70,000 grayscale images of clothing items.

Each image has a resolution of:

28 × 28 pixels

There are 10 different classes:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

The dataset is divided into:

- 60,000 training images
- 10,000 test images

## Model Architecture

The model consists of three convolutional blocks followed by
fully connected layers.

### CNN Feature Extraction

Conv2D (32 filters, 3×3)
→ Batch Normalization
→ ReLU
→ MaxPooling
→ Dropout

Conv2D (64 filters, 3×3)
→ Batch Normalization
→ ReLU
→ MaxPooling
→ Dropout

Conv2D (128 filters, 3×3)
→ Batch Normalization
→ ReLU
→ MaxPooling
→ Dropout

### ANN Classification

Flatten
→ Dense (128 neurons)
→ Batch Normalization
→ Dropout
→ Dense (10 neurons)
→ Softmax

## Results

Test Loss:

0.2687

Test Accuracy:

90.21%

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pillow
- Streamlit

## Project Structure

MNIST-Project/

├── models/

│   ├── class_names.json

│   ├── fashion_mnist_cnn_ann.keras

│   ├── test_results.json

│   └── training_history.json

├── utils/

│   └── preprocessing.py

├── app.py

├── train.py

├── requirements.txt

├── README.md

└── .gitignore

## Running the Application

Create and activate the virtual environment:

python -m venv venv

Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

The application will open in your browser.

## Features

- Upload clothing images
- Automatic image preprocessing
- Fashion-MNIST classification
- Predicted class
- Prediction confidence
- Class probability distribution
- CNN + ANN architecture display

## Model Input

The trained model expects:

28 × 28 × 1

The uploaded image is:

1. Converted to grayscale
2. Resized to 28 × 28
3. Normalized to [0, 1]
4. Converted to the required CNN input shape

## Dataset

Fashion-MNIST dataset provided by Zalando Research.

## Author

Ritabrata Mandal

B.Tech Computer Science & Engineering
NIT Durgapur