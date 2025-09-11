# Unsupervised Anomaly Detection for Aluminum Coil Manufacturing

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

Supervised learning models for quality control are ineffective at detecting novel defects and require a large, constantly updated dataset of labeled examples. Novelis needed a more agile and comprehensive solution to maintain its high-quality standards.

## Solution Approach

Our team adopted a parallel approach, exploring three different unsupervised architectures:
1.  **Variational Autoencoder (VAE)**
2.  **Generative Adversarial Network (GAN)**
3.  **Custom Transformer-based Model**

This document focuses on the design and performance of the Transformer model I developed, which proved to be the most successful.

## My Solution: A Custom Transformer Architecture for Anomaly Detection

I developed a transformer architecture from the ground up, specifically designed for the constraints of industrial anomaly detection. While transformers are known for high performance, their computational cost can make inference slow. My architecture incorporated several key innovations to manage this trade-off and achieve good results.

### Architecture Overview
The model uses an encoder to build a latent representation of input images of aluminum coils, and a decoder to loosely reconstruct them. Anomalies are identified by quantifying the deviation between encoder and decoder layer activations (reconstruction error), under the premise that anomalous regions will reconstruct more poorly than non-anomalous regions.

<img src="./images/transformer_figure.png" width="920" alt="A flow diagram of the transformer architecture showing the encoder, the bottleneck, and the decoder">

*Figure 1: Transformer flowchart showing encoder, bottleneck and decoder regions.*  

### Tech Stack:
*   **Languages:** Python
*   **ML Frameworks:** PyTorch, Scikit-learn
*   **Cloud:** Microsoft Azure ML, Azure Blob Storage
*   **Visualization:** Matplotlib, Seaborn
*   **Version Control:** Git, GitHub

### Goal:
Thus the goal of our project was to create an unsupervised ML model for anomaly (defect) detection, with high enough recall (around 95%) and specificity that it could be implemented in Novelis's manufacturing facilities. 

### Constraints:
We were required to deploy our solution in Microsoft's Azure ML cloud platform, on a T4 GPU (due to budget constraints and the available hardware stack at the factories). It's important to note that Azure has no in-house unsupervised anomaly detection model that could be used for our purposes, so we had to build our own models on the Azure ML platform.

### Solution:
My team decided to approach the project by each creating our own supervised ML models in parallel and then evaluating them against each other. I decided to develop a transformer architecture, since transformers are powerful for numerous applications including anomaly detection. The downside of using tranformers for anomaly detection is that inference tends to be relatively slow compared to other ML methods, but I found ways to manage this. 

### Key Features:
- An encoder composed of layers of pretrained Vision Transformers (ViT)
- A bottleneck that adds noise to the image representations, to force the decoder to ignore anomalies during image reconstruction
- A decoder utilizing linear attention to reconstruct images
- A loss function that calculates error from middle layers of the encoder and decoder, to allow for more degrees of freedom in finding solutions that minimize the loss

### Architecture:


<img src="./images/transformer_figure.png" width="920" alt="A flow diagram of the transformer architecture showing the encoder, the bottleneck, and the decoder">

*Figure 1: Transformer flowchart showing encoder, bottleneck and decoder regions.*  

My transformer model uses an encoder to build a latent representation of the image inputs and a decoder to reconstruct them. Anomalies were identified by quantifying the deviation between layers of the encoder and layers of the decoder (reconstruction error), with the expectation that anomalies would have a higher deviation.

#### *DINOv2 Pre-Trained ViT Encoder*  
To speed up processing and enhance extraction of important features, we used a pre-trained Vision Transformer (ViT) as our encoder. Pre-trained ViTs have been found to
perform well in unsupervised anomaly detection and are used frequently in such
architectures [5]. DINOv2 is a ViT published in 2023 by researchers at Meta AI [8], which
uses self-supervised learning to extract common features from images. It was trained on a
large, roughly 142 million image dataset mostly from ImageNet, with relatively high
resolution (416x416 pixels during part of training). It has 12 layers and follows a standard
transformer architecture (see encoder region of Figure 13 above), with each layer using
pre-trained, frozen attention weights to extract important features from the image inputs.
Each layer consists of a linear expansion, a non-linear GELU activation to model non-linear
features, and a linear projection back to the original dimensional space.  

#### *"Noisy" Bottleneck*  
This bottleneck has two functions. First, as in most similar architectures, the
bottleneck projects the patch embedding into a small latent space. But secondly, it is
designed to add “noise” to the image representations in order to force the model to learn
to transform anomalous features of the embeddings back into their original “normal”
representations. In this way, the model disrupts the accuracy of anomaly reconstructions
by teaching it during training not to retain those regions. To accomplish this, we used a high
level of dropout (30% dropout before each of the linear layers of the bottleneck).

#### *"Linear Self-Attention in the ViT Decoder*  
The decoder section of the model is composed of 8 ViT layers built as [x/x/x], which utilize a "linear" self-attention mechanism. Self-attention describes [], and the "linear" aspect here (as coined by Liu et al.) refers to a less computationally complex, more unfocused attention mechanism, characterized by [what makes it so the linear attn eqn can be recast?]. 
In their paper for their "Dinomaly" model, Liu et al. describe this attention mechanism as being particularly well-suited for industrial anomaly detection, because its lower computation speeds up inference (which is often critical in industrial applications), and the unfocused attention could help prevent highly faithful reconstructions of anomalies, which can be a problem in transformers, autoencoders, and other ML algorithms used for anomaly detection. 
#### *Unique Loss Function*  

### Results:  

<img width="918" height="450" alt="image" src="https://github.com/user-attachments/assets/4711700c-e097-4c85-b5e1-e0094906615a" />  

<img width="862" height="550" alt="image" src="https://github.com/user-attachments/assets/c2c52b00-5ea8-4bdc-9d8a-5c18c63f4598" />  

<img width="508" height="470" alt="image" src="https://github.com/user-attachments/assets/0225a406-92a5-4d20-94cf-e1f5bc7aa7c3" />  









*Outline*  
* Description of major parts of transformer
* Results
* Conclusion

The goal of my team’s Capstone (Practicum) project was to develop an unsupervised anomaly detection method with high enough [precision and specificity] to be implemented in our sponsoring company's manufacturing facilities (our sponsor for the project was Novelis, a leading manufacturer and recycler of aluminum). Several machine learning methods were tested; the one I focused on was a transformer architecture because they tend to be accurate and although typically slower than ML methods like auto-encoders, there have been recent advancements that have made them more practical in industrial anomaly detection settings. My transformer work involved building upon a recently published transformer architecture with class-leading high performance metrics and relatively fast inference speed. The flowchart shows the major processes:


- Add info on code, libraries, and methodology
  
When the transformer was evaluated on a test dataset using images from the same facilites as the training and validations sets, the performance was quite good:

On a separate test set created by the company sponsor, the anomaly detection recall remained high but recall for "good" images fell substantially. It was hypothesized that this was due to the transformer not being trained on a wide enough variety of video feed images from the manufacturing facilities. Since the transformer is only trained on "good" images, it interprets any deviation from the images it's been trained on as anomalous. Therefore it is crucial that the training set be representative of all types of "good" images. 

Despite the decreased recall of good images, the sponsor company still believed that the promising enough to develop further. Before being passed off to the sponsor, my code was prepared for implementation by preparing training, validation and inference scripts with clearly written comments and instructions (unfortunately, the code files cannot be shared as they are property of the sponsor company). 
