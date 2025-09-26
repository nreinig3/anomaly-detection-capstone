# Unsupervised Anomaly Detection for Aluminum Coil Manufacturing

**Capstone (Practicum) Project for Master of Science in Computational Analytics | Georgia Tech**

**Sponsor:** Novelis, a global leader in low-carbon aluminum manufacturing.

<img width="1121" height="426" alt="image" src="https://github.com/user-attachments/assets/42278be9-3e71-4ac6-86c8-1a7702a32550" />

*Image of a generic metal manufacturing facility, provided by Unsplash (used under Unsplash license)*


## Overview

This project developed an unsupervised deep learning model to identify defects in aluminum roll production for Novelis. The goal was to overcome the limitations of their supervised system, which struggled with new defect types and required extensive labeled data.

**My Key Achievement:**
*   **Solely architected and implemented a novel Transformer-based model** that achieved the project's target of **>95% recall** in detecting anomalous defects without using labeled training data.
*   Deployed and evaluated the model on **Microsoft Azure ML** within strict computational constraints (T4 GPU).
*   Delivered a scalable solution with the potential to significantly reduce manual quality control labor.

*(Note: My teammates developed a Variational Autoencoder (VAE) and a Generative Adversarial Network (GAN) model in parallel. My Transformer architecture outperformed both alternatives in our evalatuation metrics, but is likely somewhat slower in inference speed.)*
  
## The Problem

Supervised learning models for quality control are ineffective at detecting novel defects and require a large, constantly updated dataset of labeled examples. Novelis wanted to explore a more agile and comprehensive solution to maintain its high-quality standards.

## Solution Approach

Our team adopted a parallel approach, exploring three different unsupervised architectures:
1.  **Variational Autoencoder (VAE)**
2.  **Generative Adversarial Network (GAN)**
3.  **Custom Transformer-based Model**

This document focuses on the design and performance of the Transformer model I developed, which proved to be the most successful.

## My Solution: A Custom Transformer Architecture for Anomaly Detection

I developed a transformer architecture from the ground up, specifically designed for the constraints of industrial anomaly detection. While transformers are known for high performance, their computational cost can make inference slow. My architecture incorporated several key innovations to manage this trade-off and achieve good results.

### Architecture Overview
The model uses an encoder to create a latent representation of input images, a bottleneck to compress this representation and inject noise, and a decoder to reconstruct the images. Anomalies are identified by quantifying the deviation between encoder and decoder layer activations (reconstruction error), based on the premise that anomalous regions will reconstruct less accurately.

<img src="./images/transformer_figure.png" width="920" alt="A flow diagram of the transformer architecture showing the encoder, the bottleneck, and the decoder">

*Figure 1. Transformer flowchart showing encoder, bottleneck and decoder regions (click to enlarge).*  

### Key Technical Features

1.  **DINOv2 Pre-Trained ViT Encoder:**
    *   To enhance feature extraction, I used a pre-trained Vision Transformer (ViT) as a frozen encoder. Specifically, I employed **DINOv2** (Meta AI, 2023), which was self-supervised on 142M images.
    *   **Rationale:** Using a pre-trained, frozen encoder has been shown to significantly improve training efficiency and feature quality compared to training from scratch, when using a small or moderately-sized training dataset (<5000 images).

2.  **A "Noisy" Bottleneck for Anomaly Suppression:**
    *   The bottleneck projects patch embeddings into a latent space but is also designed to disrupt anomalous feature reconstruction.
    *   **Implementation:** I introduced a high level of dropout (**30% before each linear layer**) within the bottleneck. This "noise" forces the decoder to learn to reconstruct only the underlying "normal" patterns, effectively teaching it to ignore anomalies during the reconstruction process.
  
3.  **Decoder with Linear Self-Attention:**
    *   The decoder is composed of 8 ViT layers utilizing a **linear self-attention** mechanism.
    *   **Rationale:** Based on the work of Liu et al. ("Dinomaly"), linear attention is less computationally complex than standard self-attention, **significantly speeding up inference**. Furthermore, its "unfocused" nature helps prevent the model from achieving overly faithful reconstructions of anomalies, a common problem in other architectures.

4.  **A Unique Loss Function for Robust Training:**
    *   Instead of a simple pixel-wise reconstruction loss (e.g., MSE) calculated from the original and reconstructed images, the loss function calculates deviation from **multiple middle layers of the encoder and decoder**.
    *   **Rationale:** This provides more degrees of freedom for the model to minimize loss, leading to more stable training and better convergence on a robust solution for identifying anomalies.
  
### Tech Stack:
*   **Languages:** Python
*   **ML Frameworks:** PyTorch, Scikit-learn
*   **Model Architecture:** Custom Transformer based on DINOv2 (ViT)
*   **Cloud:** Microsoft Azure ML, Azure Blob Storage
*   **Visualization:** Matplotlib, Seaborn
*   **Version Control:** Git, GitHub

## Results

My transformer model successfully **met the project's primary goal, achieving ≥ 95% recall** on Novelis's holdout test set (Test Set #2), outperforming parallel VAE and GAN approaches. This demonstrates the strong potential of a carefully engineered transformer for identifying novel defects in an industrial setting.

**Generalization Challenge:**
The model performed well on an internal set (Test Set #1), achieving 83.3% recall for defects and 91.2% for good images. However, its performance on Novelis's separate holdout set (Test Set #2) revealed the **critical importance of data diversity**: while defect recall remained high (95%), the false positive rate increased significantly. This indicates that the training data, while accurate, was probably not fully representative of the natural variation in "good" images across different manufacturing conditions. Because of that, the model incorrectly flagged acceptable images as anomalies, highlighting the importance of maximizing diversity of the "good" class during training in order to minimize false positives in production.

**Conclusion & Handoff:**
Despite this challenge, the sponsor found the results **highly promising for future development**. To facilitate a smooth transition, I prepared the code for implementation by creating robust training, validation, and inference scripts, all thoroughly documented with clear comments and instructions.

<img width="918" height="450" alt="image" src="https://github.com/user-attachments/assets/4711700c-e097-4c85-b5e1-e0094906615a" />  

*Figure 2. Anomaly score (reconstruction error) versus image class for the Transformer model on the internal test set (Test Set #1). Generally higher anomaly scores for the 'defect' class indicate the model's effectiveness, though some overlap is present (16.7% misclassified).*

<img width="508" height="470" alt="image" src="https://github.com/user-attachments/assets/0225a406-92a5-4d20-94cf-e1f5bc7aa7c3" />  

*Figure 3. Receiver Operating Characteristic (ROC) curve for the transformer model on Test Set #1, demonstrating excellent performance (AUC = 0.96).*


---
*Please note that this project was completed under a confidentiality agreement with Novelis, therefore the code and proprietary data are not available in this repository. The purpose of this document is to outline the architectural approach and technical reasoning behind the solution.*
