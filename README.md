# Semantic Segmentation of Sonography Images: Carotid Artery Detection  

**Author**: 周佳萱  
**Course**: Deep Learning in Medical Image Analysis  
**Date**: 2024/12/08  

---

## Overview  

This project focuses on segmenting carotid arteries in sonography images using semantic segmentation models. The dataset consists of sonography images captured from volunteers' left and right necks. Several radiologists labeled the carotid artery areas for training and testing.  

### Dataset  
- **Training Data**:  
  - 300 sonography images (100 frames per volunteer from 3 volunteers).  
- **Test Data**:  
  - 100 sonography images from a separate volunteer.  

---

## Implementation  

### Models and Comparisons  

| Model         | Epochs | Learning Rate | Optimizer | Scheduler   | Data Transformation | Parameters  |  Accuraacy  |
|---------------|--------|---------------|-----------|-------------|----------------------|-------------| -------------| 
| **FCN-8s**    | 40     | 1e-3          | AdamW     | OneCycleLR  | No                   | 14,717,590  |  0.80183
| **U-Net**     | 40     | 1e-3          | AdamW     | OneCycleLR  | Yes                  | 31,030,850  |  0.94425
| **ResUnet**   | 12     | 1e-3          | AdamW     | OneCycleLR  | No                   | 13,043,074  |  0.92066
| **ResUnet++** | 10     | 1e-3          | AdamW     | OneCycleLR  | No                   | 14,482,597  |  0.93530

---

### Selected Model  

#### **U-Net**:  
- **Why U-Net?**  
  - Simpler architecture, making it easier to train on small datasets.  
  - Performs better on Kaggle public leaderboard compared to deeper models (ResUnet, ResUnet++).  
  - Avoided overfitting despite small dataset size.  

#### **Data Transformations Applied**:  
  - Resize to ensure uniform input dimensions.  
  - Random brightness and contrast adjustments to simulate lighting variations.  
  - Gaussian noise to obscure background details and emphasize vessels.  
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance local contrast.  
  - Normalization and conversion to tensors.  

  Note: Horizontal flipping was avoided as it alters blood flow direction, negatively affecting results.  


---

### Challenges and Solutions  

1. **Overfitting in Deep Models**:  
   - ResUnet and ResUnet++ models overfit quickly due to their depth and complexity.  
   - U-Net was less prone to overfitting, making it more suitable for small datasets.  

2. **Data Augmentation**:  
   - Improved U-Net performance using transformations like CLAHE and Gaussian noise.  
   - Transformations were less effective for ResUnet/ResUnet++ due to their inherent complexity.  

---

### Model Architectures  

#### **FCN-8s**:  
- **Encoder**:  
  - Blocks of 2-3 convolutional layers with max-pooling.  
  - Example:  
    - Block 1: `Input → Conv (3→64) → Conv (64→64) → MaxPool (2,2)`  
    - Block 5: `512 → 512 → 512 → MaxPool (2,2)`  

- **Decoder**:  
  - 1x1 convolutions followed by upsampling and skip connections.  
  - Example:  
    - Block 5 output is upsampled and added to Block 4’s output.  
    - Final output upsampled 8x using transpose convolution.  

---

#### **U-Net**:  
- **Contracting Path (Encoder)**:  
  - Blocks of 2 convolutional layers with ReLU activation and max-pooling.  
  - Feature maps grow deeper: `Input (3) → Block 1 (64) → Block 2 (128) → Block 3 (256) → Block 4 (512)`.  

- **Bottleneck**:  
  - Two 3x3 convolutions followed by upsampling with transpose convolution.  

- **Expanding Path (Decoder)**:  
  - Transpose convolutions for upsampling.  
  - Features from corresponding encoder layers are concatenated via skip connections.  
  - Final feature maps are reduced: `1024 → 512 → 256 → 128 → 64`.  
