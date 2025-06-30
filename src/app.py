import tkinter as tk
from tkinter import Canvas, Label, Button
from PIL import ImageGrab
import numpy as np
from src.utils import preprocess_image
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image, ImageDraw 

import matplotlib
matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt

CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

class HandwritingRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwriting Recognizer")

        # Set the canvas size
        self.canvas_size = 280

        # Canvas for drawing
        self.canvas = Canvas(self.root, width=280, height=280, bg="white", cursor="cross", highlightthickness=2, highlightbackground="black")

        self.canvas.pack()

        # Label to display prediction
        self.label = Label(self.root, text="Draw a character", font=("Arial", 16))
        self.label.pack(pady=10)

        # Buttons
        Button(self.root, text="Predict", command=self.predict, font=("Arial", 14)).pack(side="left", padx=10)
        Button(self.root, text="Clear", command=self.clear_canvas, font=("Arial", 14)).pack(side="right", padx=10)

        # Canvas drawing - simple approach
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonPress-1>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_drawing)
        
        # Track if we're drawing
        self.old_x = None
        self.old_y = None

        # Create in-memory image to match canvas size
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "white")
        self.draw_interface = ImageDraw.Draw(self.image)

        # Load the pre-trained model
        self.model = load_model("src/model/emnist_model.h5")

    def paint(self, event):
        """Simple paint function"""
        print(f"Paint called at ({event.x}, {event.y})")  # Debug line
        
        # Draw on canvas with a simple approach
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                  width=5, fill='black', capstyle='round', smooth='true')
            print(f"Drew line from ({self.old_x}, {self.old_y}) to ({event.x}, {event.y})")  # Debug
            # Also draw on the PIL image
            self.draw_interface.line([self.old_x, self.old_y, event.x, event.y], 
                                   fill='black', width=5)
        else:
            # First click - draw a dot
            self.canvas.create_oval(event.x-2, event.y-2, event.x+2, event.y+2, 
                                  fill='black', outline='black')
            print(f"Drew dot at ({event.x}, {event.y})")  # Debug
            self.draw_interface.ellipse([event.x-2, event.y-2, event.x+2, event.y+2], 
                                      fill='black')
        
        self.old_x = event.x
        self.old_y = event.y

    def reset_drawing(self, event):
        """Reset drawing coordinates"""
        self.old_x, self.old_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "white")
        self.draw_interface = ImageDraw.Draw(self.image)
        self.label.config(text="Draw a character")
        # Reset drawing position
        self.old_x = None
        self.old_y = None


    def predict(self):
        self.image.show()
        image = preprocess_image(self.image)

       
        # Show the preprocessed image for debugging
        plt.imshow(image.reshape(28, 28), cmap='gray')
        plt.title("Preprocessed Image")
        plt.axis('off')
        plt.show()


        prediction = self.model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)
        predicted_char = CLASSES[predicted_class]

        print(f"Predicted: {predicted_char} with confidence {confidence:.2%}")
        self.label.config(text=f"Prediction: {predicted_char} ({confidence:.2%})")



    # def predict(self):
    #     x = self.root.winfo_rootx() + self.canvas.winfo_x()
    #     y = self.root.winfo_rooty() + self.canvas.winfo_y()
    #     x1 = x + self.canvas.winfo_width()
    #     y1 = y + self.canvas.winfo_height()

    #     # Capture the canvas as an image
    #     image = ImageGrab.grab(bbox=(x, y, x1, y1))
    #     image = preprocess_image(image)

    #     # Predict the character
    #     prediction = self.model.predict(image)
    #     predicted_class = np.argmax(prediction, axis=1)[0]
    #     predicted_char = CLASSES[predicted_class]

    #     # Update the label with the prediction
    #     confidence = np.max(prediction)
    #     self.label.config(text=f"Prediction: {predicted_char} ({confidence:.2%})")
    #     print(f"Predicted: {predicted_char} with confidence {confidence:.2%}")


if __name__ == "__main__":
    root = tk.Tk()
    app = HandwritingRecognizerApp(root)
    root.mainloop()
