# Insect Species Classification using Deep Learning

A deep learning project for **multi-class insect species classification** using **TensorFlow** and **MobileNetV2** transfer learning.  
This project was developed in a Kaggle environment and achieved:

- **98% Training Accuracy**
- **91% Validation Accuracy**
- **0.914 Kaggle Score**

---

## Project Overview

This notebook builds an image classification pipeline to identify different insect species from images.

The workflow includes:

- Data loading & preprocessing
- Exploratory Data Analysis (EDA)
- Image visualization
- Data augmentation
- Transfer learning with MobileNetV2
- Fine-tuning pretrained layers
- Model training & evaluation
- Prediction generation for test images

---

## Tech Stack

- Python
- TensorFlow / Keras
- MobileNetV2
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Dataset Structure

```bash
dataset/
│
├── train/
│     └──train.csv
├── test/
└── sample_submission.csv
```

---

## Exploratory Data Analysis

The notebook includes:

- Species distribution analysis
- Class imbalance visualization
- Sample image visualization
- Dataset inspection using Pandas

---

## Model Architecture

### Base Model

- **MobileNetV2** pretrained on ImageNet

### Custom Layers

- Global Average Pooling
- Dense Layer (512 units)
- Dropout (0.5)
- Softmax Output Layer

---

## Training Configuration

| Parameter               | Value                    |
| ----------------------- | ------------------------ |
| Image Size              | 224x224                  |
| Batch Size              | 32                       |
| Initial Epochs          | 20                       |
| Fine-tuning Epochs      | 50                       |
| Optimizer               | AdamW                    |
| Loss Function           | Categorical Crossentropy |
| Learning Rate Scheduler | Cosine Annealing         |

---

## Data Augmentation

The model uses extensive augmentation techniques:

- Rotation
- Zoom
- Width & height shifts
- Horizontal & vertical flips
- Brightness adjustment
- Shearing

This improves generalization and reduces overfitting.

---

## Results

| Metric                        | Score |
| ----------------------------- | ----- |
| Training Accuracy             | 98%   |
| Validation Accuracy           | 91%   |
| Kaggle Score (Unseen Dataset) | 0.914 |

---

## How to Run

### Clone Repository

```bash
git clone https://github.com/your-username/insect-species-classification.git
cd insect-species-classification
```

### Install Requirements

```txt
tensorflow
numpy
pandas
matplotlib
opencv-python
scikit-learn
```

### Run Notebook

```bash
jupyter notebook kaggle-2.ipynb
```

---

## Model Export

The trained model is saved as:

```bash
insect_classifier.h5
```

---

## Sample Workflow

1. Load dataset
2. Visualize species distribution
3. Apply preprocessing & augmentation
4. Train MobileNetV2 classifier
5. Fine-tune pretrained layers
6. Generate predictions

---

## Contributing

Contributions are welcome!  
Feel free to fork the repository and submit pull requests.
