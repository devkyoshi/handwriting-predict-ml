# Handwriting Reader ML

A machine learning application that recognizes handwritten characters using a Convolutional Neural Network (CNN) trained on the EMNIST (Extended MNIST) dataset. The application features a graphical user interface built with Tkinter that allows users to draw characters and get real-time predictions.

## Features

- 🎨 **Interactive Drawing Canvas**: Draw characters directly in the application
- 🧠 **CNN-based Recognition**: Uses a trained convolutional neural network for character recognition
- 📊 **Real-time Predictions**: Get instant predictions with confidence scores
- 🔤 **Multi-character Support**: Recognizes digits (0-9) and letters (A-Z, a-z)
- 📈 **Visual Feedback**: Shows preprocessed images for debugging and understanding

## Dataset

The project uses the **EMNIST Balanced** dataset, which contains:
- 47 balanced classes (digits 0-9 and letters A-Z, a-z with some exclusions)
- 131,600 training samples
- 21,800 test samples
- 28x28 grayscale images

**Supported Characters**: `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt`

## Project Structure

```
handwriting-reader-ml/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── app.py              # Main GUI application
│   ├── train_emnist.py     # Model training script
│   ├── utils.py            # Image preprocessing utilities
│   ├── emnist/
│   │   └── emnist-balanced.mat  # EMNIST dataset
│   └── model/
│       └── emnist_model.h5      # Trained model
```

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Pillow (PIL)
- Tkinter (usually included with Python)
- NumPy
- SciPy
- Matplotlib

## Installation

1. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd handwriting-reader-ml
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the EMNIST dataset**:
   - Download `emnist-balanced.mat` from the [EMNIST website](https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist)
   - Place it in `src/emnist/emnist-balanced.mat`

## Usage

### Option 1: Use Pre-trained Model (Quick Start)

If you have a pre-trained model (`src/model/emnist_model.h5`), you can directly run the application:

```bash
python src/app.py
```

### Option 2: Train Your Own Model

1. **Train the model** (this may take several minutes):
   ```bash
   python src/train_emnist.py
   ```
   
   This will:
   - Load and preprocess the EMNIST dataset
   - Train a CNN model for 10 epochs
   - Save the trained model as `model/emnist_model.h5`

2. **Run the application**:
   ```bash
   python src/app.py
   ```

## How to Use the Application

1. **Launch the application** - A window with a drawing canvas will appear
2. **Draw a character** - Use your mouse to draw a digit or letter on the white canvas
3. **Get prediction** - Click the "Predict" button to see the model's prediction
4. **Clear canvas** - Click "Clear" to erase the canvas and draw a new character

### Tips for Better Recognition

- Draw characters clearly and at a reasonable size
- Use the full canvas space
- Make sure characters are well-centered
- For best results, draw characters similar to handwritten text

## Model Architecture

The CNN model consists of:

```
- Conv2D (32 filters, 3x3 kernel, ReLU activation)
- MaxPooling2D (2x2)
- Conv2D (64 filters, 3x3 kernel, ReLU activation)  
- MaxPooling2D (2x2)
- Flatten
- Dense (128 neurons, ReLU activation)
- Dropout (0.5)
- Dense (47 neurons, softmax activation)
```

**Training Configuration**:
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Epochs: 10
- Batch Size: 128

## Image Preprocessing Pipeline

The application preprocesses drawn images through several steps:

1. **Convert to grayscale**
2. **Resize to 28x28 pixels**
3. **Invert colors** (white background → black, black text → white)
4. **Normalize pixel values** to [0, 1]
5. **Reshape for model input** (1, 28, 28, 1)

## Troubleshooting

### Common Issues

1. **"No module named 'tensorflow'"**
   ```bash
   pip install tensorflow
   ```

2. **"No module named 'PIL'"**
   ```bash
   pip install pillow
   ```

3. **"FileNotFoundError: emnist-balanced.mat"**
   - Make sure you've downloaded the EMNIST dataset
   - Verify the file is placed in `src/emnist/emnist-balanced.mat`

4. **"FileNotFoundError: emnist_model.h5"**
   - Run the training script first: `python src/train_emnist.py`
   - Or ensure you have a pre-trained model in `src/model/`

5. **Poor recognition accuracy**
   - Try drawing characters more clearly
   - Ensure characters are centered and use most of the canvas
   - Check that the model has been trained properly

### Performance Notes

- Training the model may take 10-30 minutes depending on your hardware
- For faster training, consider using a GPU with CUDA support
- The application works best with clear, well-drawn characters

## Contributing

Feel free to contribute by:
- Improving the model architecture
- Adding support for more character classes
- Enhancing the user interface
- Optimizing preprocessing steps
- Adding more robust error handling

## License

This project is for educational purposes. Please respect the EMNIST dataset license terms.

## Acknowledgments

- **EMNIST Dataset**: Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters.
- **TensorFlow/Keras**: For providing the deep learning framework
- **Python Community**: For the excellent libraries that make this project possible

