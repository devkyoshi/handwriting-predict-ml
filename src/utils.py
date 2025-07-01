import numpy as np
from PIL import ImageOps


def preprocess_image(image):
    image = image.convert("L")
    image = image.resize((28, 28))
    image = ImageOps.invert(image)

    image_array = np.array(image).astype("float32") / 255.0
    image_array[image_array < 0.2] = 0.0

    return image_array.reshape(1, 28, 28, 1)

