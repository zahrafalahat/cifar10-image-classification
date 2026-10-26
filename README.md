# cifar10-image-classification
# CIFAR-10 Image Classification with MLP and CNN

Image classification on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset using Keras / TensorFlow. The project compares two approaches on the same data: a fully connected network (MLP) and a Convolutional Neural Network (CNN), and includes a small inference snippet to classify your own images.

## Dataset

CIFAR-10 contains 60,000 color images of size 32×32 in 10 classes:

| Split | Images | Shape |
|-------|--------|-------|
| Train | 50,000 | (50000, 32, 32, 3) |
| Test  | 10,000 | (10000, 32, 32, 3) |

**Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

The notebook loads the data from a local `cifar-10-python.npz` file with the keys `x_train`, `y_train`, `x_test`, `y_test`.

## Project Workflow

1. **Load data** from the `.npz` file.
2. **Preprocess**: normalize pixel values to the range [0, 1] by dividing by 255.
3. **Visualize** a sample image with Matplotlib.
4. **Model 1 – MLP**: flatten each image to a 3072-length vector and train a dense network.
5. **Model 2 – CNN**: train on the original 32×32×3 images so spatial structure is preserved.
6. **Predict on custom images** using OpenCV for loading and resizing.

## Models

### MLP (baseline)

```
Input (3072)
→ Dense(256) + ReLU
→ Dense(128) + ReLU
→ Dense(64)  + ReLU
→ Dense(10)  + Softmax
```

- Optimizer: Adam
- Loss: sparse categorical crossentropy
- 10 epochs, batch size 32, 20% validation split

### CNN

```
Input (32, 32, 3)
→ Conv2D(32, 3×3) + ReLU
→ MaxPooling2D(2×2)
→ Flatten
→ Dense(128) + ReLU
→ Dense(10) + Softmax
```

- Optimizer: Adam
- Loss: sparse categorical crossentropy
- 50 epochs, batch size 128

## Results

| Model | Epochs | Training accuracy | Validation accuracy |
|-------|--------|-------------------|---------------------|
| MLP   | 10     | ~49%              | ~48%                |
| CNN   | 50     | ~96%              | not measured yet    |

The CNN clearly outperforms the MLP on training data, which shows why convolutional layers suit image tasks: they exploit local spatial patterns that flattening throws away.

> **Note:** The CNN's ~96% is *training* accuracy only. With 50 epochs and no regularization, it is likely overfitting. Evaluating on the 10,000 test images (see "Future Improvements") is needed to get its true generalization performance.

## Predicting on Your Own Image

```python
import cv2
import numpy as np

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

img = cv2.imread('horse.png')
img = cv2.resize(img, (32, 32))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img / 255.0
img = img.reshape(1, 32, 32, 3)

pred = CNN.predict(img)
print(class_names[np.argmax(pred)])
```

For the MLP, reshape the image to `(1, 3072)` instead.

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install numpy matplotlib tensorflow opencv-python jupyter
```

Place `cifar-10-python.npz` in the project folder, then run:

```bash
jupyter notebook cifar-10.ipynb
```

## Project Structure

```
.
├── cifar-10.ipynb          # Full code: preprocessing, MLP, CNN, inference
├── cifar-10-python.npz     # Dataset (not included, see Dataset section)
└── README.md
```

## Future Improvements

- Evaluate both models on the test set with `model.evaluate(test_image_normal, test_labels)`.
- Add a validation split to the CNN and use early stopping to reduce overfitting.
- Deepen the CNN (more Conv + Pooling blocks) and add Dropout and Batch Normalization.
- Apply data augmentation (flips, rotations, shifts).
- Add a confusion matrix and per-class accuracy.
- Plot training/validation loss and accuracy curves.
- Save and load the trained model (`model.save(...)`).

## Tech Stack

Python · TensorFlow / Keras · NumPy · Matplotlib · OpenCV

## Author

**Hasti Falahat**
