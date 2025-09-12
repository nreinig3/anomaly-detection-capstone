# Unsupervised Anomaly Detection for Aluminum Coil Manufacturing

<img width="1118" height="498" alt="image" src="https://github.com/user-attachments/assets/f4c98a76-b490-4f80-a478-4cc329f7ccde" />

**Capstone (Practicum) project for Master of Science in Computational Analytics | Georgia Tech**

**Sponsor:** Novelis, a global leader in low-carbon aluminum manufacturing.


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

*Figure 1: Transformer flowchart showing encoder, bottleneck and decoder regions.*  

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

My transformer model successfully **met the project's primary goal, achieving &ge; 95% recall** in detecting anomalous defects on our internal test set. It outperformed the parallel VAE and GAN approaches in identifying novel defect types, demonstrating the viability of a carefully engineered transformer for high-accuracy industrial anomaly detection.

**Generalization Challenge & Insight:**
Performance was also evaluated on a separate, more challenging holdout test set provided by the sponsor. While **anomaly recall remained high, the false positive rate increased**; many known "good" images were incorrectly flagged as anomalous. This suggests the model's training data, while accurate, lacked the full diversity of "good" images found across different manufacturing conditions. Since the model learns to flag any deviation from its training set as an anomaly, a truly representative dataset is critical for minimizing false positives.

**Conclusion & Handoff:**
Despite this challenge, the sponsor found the results **highly promising for future development**. To facilitate a smooth transition, I prepared the code for implementation by creating robust training, validation, and inference scripts, all thoroughly documented with clear comments and instructions.

<img width="918" height="450" alt="image" src="https://github.com/user-attachments/assets/4711700c-e097-4c85-b5e1-e0094906615a" />  

<img width="862" height="550" alt="image" src="https://github.com/user-attachments/assets/c2c52b00-5ea8-4bdc-9d8a-5c18c63f4598" />  

<img width="508" height="470" alt="image" src="https://github.com/user-attachments/assets/0225a406-92a5-4d20-94cf-e1f5bc7aa7c3" />  

---
*This project was completed under a confidentiality agreement with Novelis. The code and proprietary data are not available in this repository. This document outlines the architectural approach and technical reasoning behind the solution.*












*Outline*  
* Description of major parts of transformer
* Results
* Conclusion

The goal of my team’s Capstone (Practicum) project was to develop an unsupervised anomaly detection method with high enough [precision and specificity] to be implemented in our sponsoring company's manufacturing facilities (our sponsor for the project was Novelis, a leading manufacturer and recycler of aluminum). Several machine learning methods were tested; the one I focused on was a transformer architecture because they tend to be accurate and although typically slower than ML methods like auto-encoders, there have been recent advancements that have made them more practical in industrial anomaly detection settings. My transformer work involved building upon a recently published transformer architecture with class-leading high performance metrics and relatively fast inference speed. The flowchart shows the major processes:


- Add info on code, libraries, and methodology
  
When the transformer was evaluated on a test dataset using images from the same facilites as the training and validations sets, the performance was quite good:

On a separate test set created by the company sponsor, the anomaly detection recall remained high but recall for "good" images fell substantially. It was hypothesized that this was due to the transformer not being trained on a wide enough variety of video feed images from the manufacturing facilities. Since the transformer is only trained on "good" images, it interprets any deviation from the images it's been trained on as anomalous. Therefore it is crucial that the training set be representative of all types of "good" images. 

Despite the decreased recall of good images, the sponsor company still believed that the promising enough to develop further. Before being passed off to the sponsor, my code was prepared for implementation by preparing training, validation and inference scripts with clearly written comments and instructions (unfortunately, the code files cannot be shared as they are property of the sponsor company). 
