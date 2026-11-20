# 🌽 AI-Powered Corn Leaf Disease Detection System

An intelligent web-based application that detects and classifies corn leaf diseases using Machine Learning and Computer Vision techniques. The system allows users to upload a corn leaf image and instantly receive disease predictions, confidence scores, and disease information through an interactive Streamlit interface.

## 🚀 Live Demo

🔗 https://cornleaf-disease-detection.streamlit.app/

---

# 📌 Project Overview

Corn is one of the most important crops worldwide, and diseases affecting corn leaves can significantly reduce yield and quality.

This project provides an automated solution for corn disease detection using image processing and machine learning techniques. The application analyzes uploaded leaf images, extracts HSV histogram features, and predicts the disease category using a hybrid PRF + SVM model.

---

# 🎯 Objectives

- Detect diseases from corn leaf images.
- Assist in early disease identification.
- Compare Machine Learning and Deep Learning approaches.
- Deploy a lightweight and interactive web application.
- Provide an easy-to-use interface for disease classification.

---

# 🦠 Diseases Detected

The application classifies images into four categories:

✅ Healthy

🟤 Common Rust

🟠 Blight

⚪ Gray Leaf Spot

---

# 📊 Dataset

**Dataset Source:**

Corn or Maize Leaf Disease Dataset (Kaggle)

**Classes:**

1. Healthy
2. Common Rust
3. Gray Leaf Spot
4. Blight

**Characteristics:**

- RGB Images
- Multiple disease categories
- Various lighting conditions
- Different disease severity levels

---

# 🧠 Machine Learning Approach

## Feature Extraction

The system uses HSV Histogram extraction.

### Why HSV?

HSV separates:

- Hue (Color)
- Saturation (Intensity)
- Value (Brightness)

Since corn leaf diseases primarily appear as color variations, HSV provides better feature representation compared to RGB.

### Histogram Configuration

```python
[8, 8, 8]
```

Resulting feature vector size:

```text
8 × 8 × 8 = 512 features
```

---

# ⚙️ Classification Pipeline

```text
Input Image
      ↓
Resize (128 × 128)
      ↓
HSV Histogram Extraction
      ↓
Random Forest
      ↓
Class Probabilities
      ↓
StandardScaler
      ↓
Support Vector Machine
      ↓
Disease Prediction
```

---

# 🔍 Why PRF + SVM?

The deployed model uses a hybrid architecture:

### Stage 1: Probabilistic Random Forest (PRF)

```python
RandomForestClassifier(n_estimators=100)
```

The Random Forest predicts probabilities for each disease class.

Example:

```text
Blight = 0.82
Common Rust = 0.10
Gray Leaf Spot = 0.05
Healthy = 0.03
```

---

### Stage 2: Support Vector Machine (SVM)

```python
SVC(
    kernel='rbf',
    C=10,
    probability=True
)
```

The probability outputs from Random Forest become input features for SVM.

This creates a more discriminative feature space and improves classification accuracy.

---

# 🤖 Deep Learning Models Evaluated

The project also involved training and comparing several Deep Learning models:

- CNN (3 Layer)
- CNN (4 Layer)
- MobileNetV2
- VGG16
- InceptionV3
- ResNet152
- EfficientNet

---

# 📈 Performance Comparison

| Model | Accuracy |
|---------|---------|
| PRF + SVM | 93.0% |
| Random Forest | 91.76% |
| SVM | 88% |
| KNN | 87% |
| CNN (3 Layer) | 92.84% |
| CNN (4 Layer) | 93.20% |
| MobileNetV2 | 92.60% |
| VGG16 | 91.41% |
| InceptionV3 | 91.39% |

---

# ❓ Why Was PRF + SVM Deployed Instead of CNN?

Although the custom CNN achieved slightly higher accuracy (93.2%), the deployed application uses the PRF + SVM model because:

- Lower computational cost
- Faster loading time
- Easier serialization using Joblib
- Lightweight deployment on Streamlit
- Comparable accuracy to CNN

This makes the deployed solution more practical while maintaining excellent performance.

---

# 🛠️ Technologies Used

### Programming Language

- Python

### Libraries

- Streamlit
- OpenCV
- NumPy
- Scikit-Learn
- Joblib
- Matplotlib
- Pillow

### Development Tools

- VS Code
- GitHub
- Kaggle
- Streamlit Community Cloud

---

# 📂 Project Structure

```text
cornleaf-disease-detection/
│
├── imgapp.py
├── corn_leaf.pkl
├── back.jpg
├── requirements.txt
├── README.md
│
└── Dataset
```

---

# 💻 Installation

Clone the repository:

```bash
git clone https://github.com/Ananyar2006/cornleaf-disease-detection.git
```

Move into the project folder:

```bash
cd cornleaf-disease-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run imgapp.py
```

---

# 🖥️ Application Features

✅ Upload corn leaf image

✅ Disease classification

✅ Confidence score display

✅ Interactive confidence graph

✅ Disease information display

✅ Responsive Streamlit interface

✅ Agricultural image-based diagnosis

---

# 📋 Disease Information Provided

### Healthy

The corn leaf appears healthy with no visible disease symptoms.

### Common Rust

Common Rust is characterized by reddish-brown pustules on leaf surfaces and can affect crop productivity.

### Blight

Blight produces elongated tan or brown lesions and may spread rapidly under favorable conditions.

### Gray Leaf Spot

Gray Leaf Spot causes rectangular gray-colored lesions that can reduce photosynthetic efficiency and yield.

---

# 🔬 Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

These metrics were used to compare ML and DL approaches objectively.

---

# 🚧 Challenges Faced

- Selecting the most effective feature extraction technique
- Comparing Machine Learning and Deep Learning methods
- Model serialization and deployment issues
- Library version compatibility during Streamlit deployment

---

# 🔮 Future Scope

- Mobile application deployment
- Real-time camera-based detection
- Multi-crop disease identification
- Drone-based agricultural monitoring
- IoT integration for smart farming

---

# 👨‍💻 Author

**Ananya R**

Artificial Intelligence & Machine Learning

Corn Leaf Disease Detection Project

---

# ⭐ Key Achievement

Successfully designed, trained, compared, and deployed a corn disease detection system using Machine Learning and Deep Learning techniques, achieving **93% classification accuracy** with a lightweight PRF + SVM deployment model.

---

# 📜 License

This project is intended for educational, research, and learning purposes.
