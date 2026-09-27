import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

from utils.preprocessing import preprocess_image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👕",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "models/fashion_mnist_cnn_ann.keras"
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

    with open("models/class_names.json", "r") as file:
        return json.load(file)


model = load_model()
class_names = load_class_names()


# ============================================================
# HEADER
# ============================================================

st.title("👕 Fashion MNIST Classifier")

st.write(
    "CNN + ANN based image classification using TensorFlow and Keras."
)

st.divider()


# ============================================================
# MODEL INFORMATION
# ============================================================

st.subheader("Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Architecture",
        value="CNN + ANN"
    )

with col2:
    st.metric(
        label="Input",
        value="28 × 28 × 1"
    )

with col3:
    st.metric(
        label="Test Accuracy",
        value="90.21%"
    )


st.divider()


# ============================================================
# UPLOAD SECTION
# ============================================================

st.subheader("📤 Upload an Image")

st.write(
    "Upload a Fashion-MNIST style clothing image."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # --------------------------------------------------------
    # IMAGE + RESULT COLUMNS
    # --------------------------------------------------------

    image_col, result_col = st.columns(
        [1, 1],
        gap="large"
    )


    # ========================================================
    # LEFT: IMAGE
    # ========================================================

    with image_col:

        st.subheader("Uploaded Image")

        st.image(
            image,
            width=400
        )

        st.caption(
            f"File: {uploaded_file.name}"
        )


    # ========================================================
    # PREPROCESS
    # ========================================================

    processed_image = preprocess_image(image)


    # ========================================================
    # PREDICTION
    # ========================================================

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = class_names[predicted_index]

    confidence = (
        float(probabilities[predicted_index])
        * 100
    )


    # ========================================================
    # RIGHT: RESULT
    # ========================================================

    with result_col:

        st.subheader("Prediction")

        st.success(
            f"Predicted Class: **{predicted_class}**"
        )

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )


        st.subheader("Class Probabilities")


        # Sort from highest to lowest
        sorted_indices = np.argsort(
            probabilities
        )[::-1]


        for index in sorted_indices:

            class_name = class_names[index]

            probability = float(
                probabilities[index]
            )

            percentage = probability * 100

            st.write(
                f"**{class_name}** — "
                f"{percentage:.2f}%"
            )

            st.progress(
                probability
            )


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

st.divider()

st.subheader("🧠 Model Architecture")

st.write(
    "The model uses three convolutional blocks followed "
    "by a fully connected ANN classification head."
)


architecture_col1, architecture_col2 = st.columns(
    [1, 1],
    gap="large"
)


with architecture_col1:

    st.markdown("### CNN Feature Extraction")

    st.info(
        """
        **Input**

        28 × 28 × 1

        ↓

        **Data Augmentation**

        ↓

        **Conv2D**
        32 filters, 3×3

        ↓

        Batch Normalization

        ↓

        ReLU

        ↓

        MaxPooling

        ↓

        Dropout 20%

        ↓

        **Conv2D**
        64 filters, 3×3

        ↓

        Batch Normalization

        ↓

        ReLU

        ↓

        MaxPooling

        ↓

        Dropout 25%

        ↓

        **Conv2D**
        128 filters, 3×3

        ↓

        Batch Normalization

        ↓

        ReLU

        ↓

        MaxPooling

        ↓

        Dropout 30%
        """
    )


with architecture_col2:

    st.markdown("### ANN Classification")

    st.success(
        """
        **Flatten**

        ↓

        **Dense**
        128 neurons

        ↓

        ReLU

        ↓

        Batch Normalization

        ↓

        Dropout 40%

        ↓

        **Dense**
        10 neurons

        ↓

        Softmax

        ↓

        **Fashion-MNIST Class**
        """
    )


# ============================================================
# CLASS LIST
# ============================================================

st.divider()

st.subheader("👗 Supported Classes")

class_col1, class_col2 = st.columns(2)

with class_col1:

    for i in range(5):

        st.write(
            f"**{i}.** {class_names[i]}"
        )


with class_col2:

    for i in range(5, 10):

        st.write(
            f"**{i}.** {class_names[i]}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Fashion MNIST Classification Project • "
    "TensorFlow + Keras + Streamlit"
)