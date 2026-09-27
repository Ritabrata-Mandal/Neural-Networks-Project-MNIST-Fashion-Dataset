from PIL import Image, ImageOps
import numpy as np


def preprocess_image(image):

    # --------------------------------------------------------
    # 1. Convert to grayscale
    # --------------------------------------------------------

    image = image.convert("L")


    # --------------------------------------------------------
    # 2. Resize to Fashion-MNIST size
    # --------------------------------------------------------

    image = image.resize(
        (28, 28),
        Image.Resampling.LANCZOS
    )


    # --------------------------------------------------------
    # 3. Convert to NumPy
    # --------------------------------------------------------

    image_array = np.array(
        image
    ).astype("float32")


    # --------------------------------------------------------
    # 4. Automatically detect inverted images
    #
    # Fashion-MNIST:
    #   background ≈ black
    #   object ≈ white/gray
    #
    # If the corners are bright, the image is probably
    # inverted, so invert it.
    # --------------------------------------------------------

    corner_pixels = np.array([
        image_array[0, 0],
        image_array[0, -1],
        image_array[-1, 0],
        image_array[-1, -1]
    ])

    corner_mean = corner_pixels.mean()

    if corner_mean > 127:

        image_array = 255.0 - image_array


    # --------------------------------------------------------
    # 5. Normalize exactly like training
    # --------------------------------------------------------

    image_array = image_array / 255.0


    # --------------------------------------------------------
    # 6. Add channel dimension
    #
    # (28, 28)
    #      ↓
    # (28, 28, 1)
    # --------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=-1
    )


    # --------------------------------------------------------
    # 7. Add batch dimension
    #
    # (28, 28, 1)
    #      ↓
    # (1, 28, 28, 1)
    # --------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    return image_array