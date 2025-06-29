import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def preprocess_image(image):
    image = image.convert("L")                 # Grayscale
    image = image.resize((28, 28))             # Resize
    # image = ImageOps.mirror(image)             # 🔄 Flip horizontally!

    # Flip horizontally if needed
    if image.size[0] > image.size[1]:
        image = ImageOps.mirror(image)

    image = ImageOps.invert(image)             # Invert to match EMNIST
    image = np.array(image).astype("float32") / 255.0
    return image.reshape(1, 28, 28, 1) 