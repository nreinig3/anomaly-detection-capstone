# Defect Detection on Coil Manufacturing Image Data

#### Capstone (Practicum) project for Master of Science in Computational Analytics degree @ Georgia Tech

Problem: 
My team's project was sponsored by Novelis, a global leader in the production of low-carbon Aluminum rolls used by beverage packaging, automotive, aerospace, and other industries. At the time of the project. Novelis was using a supervised machine learning model detect defects in its finished Aluminum rolls. But because supervised learning models aren't good at detecting novel types of defects, they wanted to know if unsupervised methods could be feasible. 

Goal: 
Thus the goal of our project was to create an unsupervised ML model for anomaly (defect) detection, with high enough recall (around 95%) and specificity that it could be implemented in Novelis's manufacturing facilities. 

Solution:
My team decided to approach the project by each creating our own supervised ML models in parallel and then evaluate them against each other. 

The goal of my team’s Capstone (Practicum) project was to develop an unsupervised anomaly detection method with high enough [precision and specificity] to be implemented in our sponsoring company's manufacturing facilities (our sponsor for the project was Novelis, a leading manufacturer and recycler of aluminum). Several machine learning methods were tested; the one I focused on was a transformer architecture because they tend to be accurate and although typically slower than ML methods like auto-encoders, there have been recent advancements that have made them more practical in industrial anomaly detection settings. My transformer work involved building upon a recently published transformer architecture with class-leading high performance metrics and relatively fast inference speed. The flowchart shows the major processes:

<img src="./images/transformer_figure.png" width="920" alt="A flow diagram of the transformer architecture showing the encoder, the bottleneck, and the decoder">

- Add info on code, libraries, and methodology
  
When the transformer was evaluated on a test dataset using images from the same facilites as the training and validations sets, the performance was quite good:

On a separate test set created by the company sponsor, the anomaly detection recall remained high but recall for "good" images fell substantially. It was hypothesized that this was due to the transformer not being trained on a wide enough variety of video feed images from the manufacturing facilities. Since the transformer is only trained on "good" images, it interprets any deviation from the images it's been trained on as anomalous. Therefore it is crucial that the training set be representative of all types of "good" images. 

Despite the decreased recall of good images, the sponsor company still believed that the promising enough to develop further. Before being passed off to the sponsor, my code was prepared for implementation by preparing training, validation and inference scripts with clearly written comments and instructions (unfortunately, the code files cannot be shared as they are property of the sponsor company). 
