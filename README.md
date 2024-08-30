# Explainable AI algorithms in Python

This project was first made for my bachelor's thesis, but later it expanded. It aims to review popular XAI algorithms and thiers implementation in Python. 






## About

This project shows explanation algorithms for three types of data:

*   Tables

In this case, dataset is 'stroke prediction dataset' from kaggle (https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). Dataset contains information on patients like age, bmi, heart diseases etc.. There are two models - LogisticRegression and RandomForestClassifier. The first one is explained using interpret library. The second one is explained by lime, shap and pdp algorithms that is implemented in scikit-learn library.

*   Images

Second dataset is 'flowers' from tensorflow.datasets. Dataset is made of 3680 images of flowers and the goal is to classify them to 5 classes - tulips, sunflowers, roses, dandelion and daisy. In this case, I used custom CNN model build in tensorflow. The explanation was made in lime, shap and integrated gradients that is implemented in saliency library.

*   Text (WIP)

Third dataset is 'sarcasm' that contains over 25 000 sentences that are clasified as 'no sarcasm' and 'sarcasm'. The model is RNN-LSTM model build in tensorflow. WIP: explanation is in progess.

## Technologies

The project was created using those libries:
*   pandas 2.2.2
*   numpy 1.26.4
*   interpret 0.6.1
*   matplotlib 3.9.0
*   scikit-learn 1.5.0
*   tensorflow 2.16.1
*   lime 0.2.0.1
*   shap 0.45.1
*   saliency 0.2.1
