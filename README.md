# Adaptive Confidence-Guided Hybrid CNN–ViT for Brain Tumor MRI Classification

A hybrid deep learning framework that combines **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)** using an **Adaptive Confidence-Guided Fusion** mechanism for multi-class brain tumor classification from MRI images.

---

## Overview

This project implements **ACGHV (Adaptive Confidence-Guided Hybrid Vision)**, a hybrid CNN–Vision Transformer framework designed to classify brain MRI images into four categories:

* Glioma
* Meningioma
* No Tumor
* Pituitary

The framework combines the ability of CNNs to capture **local spatial features** with the ability of Vision Transformers to model **global contextual relationships** across an image.

Instead of using fixed fusion weights, the proposed approach dynamically determines the contribution of each branch based on its prediction confidence.

The project also incorporates **Explainable AI (XAI)** techniques to provide visual interpretation of model predictions.

---

## Architecture

```text
                         MRI Image
                             |
              +--------------+--------------+
              |                             |
              v                             v
        CNN Branch                    ViT Branch
      EfficientNetB0                Patch Embedding
              |                             |
              v                             v
       CNN Features                  Transformer
              |                      Encoders
              |                             |
              v                             v
       CNN Prediction                 ViT Prediction
              |                             |
              +-------------+---------------+
                            |
                            v
                 Confidence Estimation
                            |
                            v
              Adaptive Confidence-Guided
                       Fusion
                            |
                            v
                  Classification Head
                            |
                            v
                   Tumor Prediction
                            |
                            v
                    XAI Analysis
                 Grad-CAM + SHAP
```

---

## Key Components

### CNN Branch

The CNN branch uses **EfficientNetB0** with ImageNet-pretrained weights to extract detailed local and spatial features from MRI images.

### Vision Transformer Branch

The Vision Transformer branch divides the input image into **16 × 16 patches** and processes them using transformer encoder layers to capture global relationships between different image regions.

### Adaptive Confidence-Guided Fusion

The framework calculates the prediction confidence of both branches and uses these confidence values to dynamically determine their contribution to the final fused representation.

The fusion weights are calculated as:

```text
C_CNN = max(softmax(CNN logits))
C_ViT = max(softmax(ViT logits))

α = C_CNN / (C_CNN + C_ViT)
β = C_ViT / (C_CNN + C_ViT)
```

The final representation is obtained using:

```text
F_fused = α × F_CNN + β × F_ViT
```

This allows the model to adaptively combine the CNN and ViT representations instead of relying on predefined fixed weights.

---

## Dataset

The implementation supports the following Kaggle datasets:

* `masoudnickparvar/brain-tumor-mri-dataset`
* `sartajbhuvaji/brain-tumor-classification-mri`

The default configuration uses the **Masoud Brain Tumor MRI Dataset**.

The classification task contains four classes:

```text
glioma
meningioma
notumor
pituitary
```

The configured dataset split contains:

```text
Training   : 4760 images
Validation :  840 images
Testing    : 1600 images
```

The data pipeline performs preprocessing, augmentation, and stratified splitting to maintain class distribution.

---

## Data Preprocessing

Input MRI images are resized to:

```text
224 × 224
```

The training pipeline applies augmentation techniques including:

* Random horizontal flipping
* Random rotation
* Random zoom
* Random translation
* Random contrast adjustment

These transformations help improve model generalization during training.

---

## Model Configuration

| Parameter                | Configuration  |
| ------------------------ | -------------- |
| Input Size               | 224 × 224      |
| CNN Backbone             | EfficientNetB0 |
| ViT Patch Size           | 16 × 16        |
| ViT Projection Dimension | 256            |
| Transformer Layers       | 6              |
| Attention Heads          | 8              |
| MLP Dimension            | 512            |
| Dropout                  | 0.1            |
| Batch Size               | 32             |
| Learning Rate            | 1 × 10⁻⁴       |
| Epochs                   | 40             |
| Classes                  | 4              |

---

## Training

The model is trained using the combined outputs of the hybrid architecture.

Auxiliary CNN and ViT classification outputs are also used during training to encourage effective learning from both individual branches.

The training configuration includes:

* Adam-based optimization
* Learning rate: `1e-4`
* Batch size: `32`
* Training epochs: `40`
* GPU acceleration recommended

---

## Evaluation

The implementation evaluates the model using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification report
* Confusion matrix
* ROC curves
* AUC
* Ablation analysis

The evaluation pipeline provides both numerical performance measurements and visual analysis of classification behavior.

---

## Ablation Study

The project evaluates multiple model configurations to investigate the contribution of the different components:

```text
CNN Only
   |
ViT Only
   |
Static Fusion
   |
Adaptive Confidence-Guided Fusion
```

This allows the effect of the hybrid architecture and adaptive fusion mechanism to be analyzed separately.

---

## Explainable AI

The project incorporates two explainability techniques.

### Grad-CAM

Grad-CAM is used to visualize important spatial regions contributing to CNN-based predictions.

This provides a visual indication of where the model focuses when making a classification.

### SHAP

SHAP is used to analyze feature contributions and provide additional insight into the model's predictions.

Together, these techniques provide complementary explanations of model behavior.

---

## Technologies

* Python
* TensorFlow / Keras
* EfficientNetB0
* Vision Transformer
* NumPy
* Pandas
* Scikit-learn
* OpenCV
* SHAP
* Matplotlib
* Seaborn
* KaggleHub
* Google Colab / GPU

---

## Repository Structure

```text
.
├── README.md
└── ACGHV_Brain_Tumor_MRI_Classification.ipynb
```

The Jupyter Notebook contains the complete implementation, including:

```text
Dataset Preparation
        ↓
Preprocessing & Augmentation
        ↓
CNN Feature Extraction
        ↓
ViT Feature Extraction
        ↓
Confidence-Guided Fusion
        ↓
Classification
        ↓
Evaluation
        ↓
Ablation Study
        ↓
Grad-CAM & SHAP
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

### 2. Open the notebook

Open:

```text
ACGHV_Brain_Tumor_MRI_Classification.ipynb
```

using **Google Colab** or **Jupyter Notebook/JupyterLab**.

### 3. Install dependencies

The notebook handles the required package installation. The main dependencies include:

```bash
pip install tensorflow kagglehub shap opencv-python-headless seaborn scikit-learn
```

### 4. Run the notebook

Execute the notebook cells sequentially.

A **GPU-enabled environment is strongly recommended** for training the hybrid CNN–ViT architecture.

---

## Authors

**Arjya Kumar Paul**
**Soumyajit Banerjee**
**Subho Pramanick**
**Snigdha Saha**

### Mentor

**Dr. Ardhendu Sarkar**
Assistant Professor, Department of Computer Science & Engineering
Institute of Engineering and Management, Kolkata

---

## Disclaimer

This project is intended for **research and educational purposes only**.

The model is not intended to serve as a standalone medical diagnostic system. Any model predictions should be interpreted and validated by qualified medical professionals using appropriate clinical procedures.
