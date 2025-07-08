# 🧠 Brain Tumor Detection Using Deep Learning with VGG16

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)

> A deep learning-based image classification project for automatic **brain tumor detection** from MRI scans using **transfer learning** with **VGG16** architecture.

---

## 📌 Table of Contents

- [🧠 Dataset](#-dataset)
- [🧰 Libraries Used](#-libraries-used)
- [🚀 Project Features](#-project-features)
- [🧠 Model Architecture](#-model-architecture)
- [📊 Evaluation Metrics](#-evaluation-metrics)
- [🧪 Example Predictions](#-example-predictions)
- [🛠️ How to Run](#️-how-to-run)
- [✅ Results](#-results)
- [🔮 Future Work](#-future-work)
- [📚 References](#-references)
- [📝 License](#-license)

---

## 🧠 Dataset

The dataset contains labeled **brain MRI images** categorized into **four classes**:

```
/Training
├── glioma
├── meningioma
├── pituitary
└── notumor

/Testing
├── glioma
├── meningioma
├── pituitary
└── notumor
```

Each folder contains JPEG images of MRI scans related to the respective tumor class.

---

## 🧰 Libraries Used

| Purpose          | Libraries               |
| ---------------- | ----------------------- |
| Deep Learning    | `TensorFlow`, `Keras`   |
| Image Processing | `Pillow (PIL)`          |
| Visualization    | `Matplotlib`, `Seaborn` |
| Numerics         | `NumPy`                 |
| Evaluation       | `Scikit-learn`          |

---

## 🚀 Project Features

- 📁 **Data loading and preprocessing**
- 🧠 **Transfer learning with VGG16**
- 🔍 **Custom classifier for 4-class output**
- 🧪 **Evaluation using classification report, confusion matrix, ROC-AUC**
- 🖼️ **Visual predictions with class label and confidence overlay**

---

## 🧠 Model Architecture

```text
Pretrained VGG16 (frozen base)
└── Flatten
└── Dense (256 units, ReLU)
└── Dropout (0.5)
└── Dense (4 units, Softmax)
```

- **Optimizer**: Adam (`lr=0.0001`)
- **Loss**: categorical_crossentropy
- **Metrics**: accuracy

---

## 📊 Evaluation Metrics

- ✅ Classification Report (precision, recall, F1-score)
- ✅ Confusion Matrix
- ✅ ROC-AUC Curve (multi-class)

---

## 🧪 Example Predictions

```python
img_path = './Testing/meningioma/Te-me_0017.jpg'
detect_and_display(img_path, model)
```

- ✅ Tumor class displayed
- ✅ Confidence score shown
- ✅ Image plotted with prediction overlay

---

## 🛠️ How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/brain-tumor-detection
cd brain-tumor-detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Prepare Dataset

Ensure your dataset is placed in the `/Training` and `/Testing` folders as shown.

### 4️⃣ Run the Notebook

```bash
jupyter notebook model.ipynb
# or open in VS Code / Colab
```

---

## ✅ Results

- 📈 High classification accuracy
- 🔬 Clear distinction between tumor types
- 🧠 Real-time predictions with visual output
- 💯 Confident tumor detection in unseen MRI scans

---

## 🔮 Future Work

- ✅ Add data augmentation for better generalization
- ✅ Try deeper models (ResNet, EfficientNet)
- ✅ Deploy using Streamlit or Flask
- ✅ Integrate **Grad-CAM** for attention visualization

---

## 📚 References

- 📊 [Kaggle Brain MRI Dataset](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
- 🧠 [VGG16 Paper](https://arxiv.org/abs/1409.1556)
- 📘 [TensorFlow Docs](https://www.tensorflow.org/)

---

## 📝 License

This project is licensed under the **MIT License**. Feel free to use and modify it for research and educational purposes.
