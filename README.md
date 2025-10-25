# Character Recognition CNN with EMNIST Dataset

A Convolutional Neural Network (CNN) implementation for recognizing handwritten alphabetic characters (A-Z) using the EMNIST Letters dataset.

## Overview

This project implements a CNN model that achieves ~94% accuracy on the EMNIST Letters dataset. The model is designed to recognize handwritten uppercase letters and can be used for custom image prediction.

## Features

- **CNN Architecture**: Custom 2-layer convolutional neural network with dropout regularization
- **EMNIST Dataset**: Trained on 88,800 training samples and tested on 14,800 test samples
- **Image Preprocessing**: Advanced preprocessing pipeline for custom images including:
  - Grayscale conversion
  - Adaptive thresholding
  - Aspect ratio preservation
  - Center-of-mass alignment
  - EMNIST-style rotation and normalization
- **Performance Visualization**: Training metrics including accuracy, loss, learning rate, and overfitting trends
- **Custom Image Testing**: Support for testing on your own handwritten character images

## Model Architecture

```
Input (28x28x1) 
    ↓
Conv2D (32 filters, 3x3, ReLU)
    ↓
MaxPool2D (2x2)
    ↓
Conv2D (64 filters, 3x3, ReLU)
    ↓
MaxPool2D (2x2)
    ↓
Dropout (0.25)
    ↓
Flatten (64 * 7 * 7)
    ↓
Dense (256, ReLU)
    ↓
Dense (26, Softmax)
```

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
opencv-python>=4.8.0
Pillow>=10.0.0
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/character-recognition-cnn.git
cd character-recognition-cnn

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model

Open `character_recognition_cnn.ipynb` in Google Colab or Jupyter Notebook and run all cells. The notebook will:

1. Download and prepare the EMNIST dataset
2. Train the CNN model for 12 epochs
3. Display training metrics and visualizations
4. Evaluate on the test set

### Testing on Custom Images

To test on your own handwritten characters:

1. Upload images to Google Drive in a folder (e.g., `A-Z/`)
2. Name files with the format: `{LETTER}_description.jpg` (e.g., `A_sample.jpg`)
3. Update the `custom_image_folder` path in the notebook
4. Run the custom testing section

**Image Requirements:**
- Format: PNG, JPG, or JPEG
- Content: Single uppercase letter
- Background: White or transparent
- Letter color: Dark (black or dark gray)

## Results

- **Training Accuracy**: 96.61%
- **Test Accuracy**: 93.86%
- **Overfitting**: ~2.5% (acceptable)

### Training Visualization

The notebook generates plots showing:
- Learning rate schedule
- Training vs. test accuracy over epochs
- Training loss progression
- Overfitting metrics (accuracy gap)

## Dataset

The EMNIST Letters dataset contains:
- 88,800 training samples
- 14,800 test samples
- 26 classes (A-Z uppercase letters)
- Image size: 28x28 grayscale

Dataset is automatically downloaded via torchvision.

## Preprocessing Pipeline

The custom image preprocessing ensures compatibility with EMNIST format:

1. Grayscale conversion
2. Background inversion (if needed)
3. Gaussian blur for noise reduction
4. Adaptive thresholding
5. Bounding box cropping
6. Aspect ratio-preserving resize
7. Center padding to 28x28
8. Center-of-mass alignment
9. 180° rotation + horizontal flip (EMNIST orientation)
10. Normalization

## Project Structure

```
character-recognition-cnn/
│
├── character_recognition_cnn.ipynb    # Main notebook
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
└── .gitignore                        # Git ignore file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- EMNIST Dataset: Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017)
- PyTorch framework
- Google Colab for providing free GPU resources

## Future Improvements

- [ ] Add data augmentation during training
- [ ] Implement learning rate scheduling
- [ ] Test with different architectures (ResNet, EfficientNet)
- [ ] Create a web interface for real-time predictions
- [ ] Extend to lowercase letters and digits
- [ ] Model quantization for deployment
- [ ] Add confusion matrix visualization
