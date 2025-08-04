# 🧠 Lung Cancer Detection using Vision Transformers

A machine learning project to detect **lung cancer types** from medical images using **Vision Transformer (ViT)** and **Swin Transformer** models.

---

## 📌 Project Objective

To accurately classify lung CT scan images into:
- **Adenocarcinoma**
- **Large Cell Carcinoma**
- **Squamous Cell Carcinoma**
- **Normal (Non-cancerous)**

Using deep learning and transformer-based architectures to assist in early and precise diagnosis.

---
## 🗂️ Project Structure
```
Lung_Cancer_Detection/
├── Trained Model Notebooks/
│   ├── vit_tiny.ipynb
│   ├── vit_large_patch16_224.ipynb
│   ├── swin_tiny_patch4_window7_224.ipynb
│   ├── swin_base_patch4_window7_224.ipynb
│   ├── maxvit_tiny_rw_224.ipynb
│   ├── deit_base_patch16_224.ipynb
│   └── beit_base_patch16_224.ipynb
│
├── README.md
├── requirements.txt

```
---
## 📂 Dataset

The dataset contains labeled **CT scan images** for four types of lung conditions:

- **adenocarcinoma**
- **large cell carcinoma**
- **squamous cell carcinoma**
- **normal (non-cancerous)**

This dataset was used to train and evaluate all models for accurate lung cancer classification.

### 📥 Download Link

👉 **[You can Download the Dataset from here or directly from kaggle website](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images?resource=download)**  


### 📁 Folder Structure
```
Data/
├── train/
│   ├── adenocarcinoma/
│   ├── large.cell.carcinoma/
│   ├── squamous.cell.carcinoma/
│   └── normal/
├── valid/
│   ├── adenocarcinoma/
│   ├── large.cell.carcinoma/
│   ├── squamous.cell.carcinoma/
│   └── normal/
└── test/
    ├── adenocarcinoma/
    ├── large.cell.carcinoma/
    ├── squamous.cell.carcinoma/
    └── normal/

```

Each folder contains CT scan images in `.jpg` or `.png` format specific to that class.

---
## 📚 Models Used

This project explores and compares various **Vision Transformer-based models** for lung cancer detection:

### ✅ Trained Models:

- **ViT (Vision Transformer) - Tiny & Large**
  - Standard transformer architecture for images.
  - Used `vit_tiny` and `vit_large_patch16_224`.

- **Swin Transformer - Tiny & Base**
  - Hierarchical vision transformer using shifted windows.
  - Models used: `swin_tiny_patch4_window7_224`, `swin_base_patch4_window7_224`.

- **MaxViT**
  - Combines convolution and self-attention in a scalable way.
  - Model used: `maxvit_tiny_rw_224`.

- **DeiT (Data-efficient Image Transformer)**
  - Optimized for training with fewer images.
  - Model used: `deit_base_patch16_224`.

- **BEiT (Bidirectional Encoder Representation from Image Transformers)**
  - Trained using masked image modeling similar to BERT.
  - Model used: `beit_base_patch16_224`.

### ⚖️ Goal:

To compare the performance of these models on lung cancer image classification and identify the most effective one for medical image analysis.

---
## 🔄 Workflow

The complete workflow of the Lung Cancer Detection project is as follows:

1. **Dataset Preparation**
   - Organized images into `train`, `valid`, and `test` directories by class
   - Preprocessed the data (resizing, normalization, etc.)

2. **Model Selection**
   - Implemented and compared multiple transformer-based models:
     - ViT (Vision Transformer)
     - Swin Transformer (tiny & base)
     - MaxViT
     - DeiT
     - BEiT

3. **Training**
   - Used Google Colab with GPU support
   - Fine-tuned pretrained models using `timm` library
   - Applied data augmentation and early stopping

4. **Evaluation**
   - Evaluated models using:
     - Accuracy
     - Precision / Recall
     - Confusion Matrix
     - Classification Report

5. **Model Comparison**
   - Compared performance of all models to identify the best one

6. **Future Plans**
   - Add Grad-CAM visualizations
   - Deploy as a Streamlit web app
   - Train on larger datasets for generalization
 ---
## 📊 Evaluation Metrics

The performance of each model was evaluated using standard classification metrics:

### 🧪 Metrics Used:

- **Accuracy**  
  Measures the overall correctness of the model.

- **Precision**  
  Indicates how many of the predicted positive cases were actually correct.  
  `Precision = TP / (TP + FP)`

- **Recall (Sensitivity)**  
  Shows how well the model captures actual positive cases.  
  `Recall = TP / (TP + FN)`

- **F1-Score**  
  Harmonic mean of Precision and Recall.  
  `F1 = 2 * (Precision * Recall) / (Precision + Recall)`

- **Confusion Matrix**  
  Visual representation of True vs. Predicted labels across all classes.

- **Classification Report**  
  Provides class-wise Precision, Recall, F1-score, and Support.

---

These metrics were calculated on the **test dataset** after training to ensure unbiased evaluation of model performance.

---
