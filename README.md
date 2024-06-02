
# Plant disease identification

##### Author: Nikas Belogolov, י"א 6

---

## Introduction

This project aims to develop a neural network-based image classifier to identify diseases in pepper and potato plants. Leveraging deep learning techniques, the model will be trained to distinguish between healthy plants and those affected by various diseases. This application is crucial for early disease detection and effective crop management, potentially leading to higher yields and reduced losses.

### Goals

1. Identify Diseases: Develop a robust model capable of identifying diseases in pepper and potato plants from images.
2. Learn Image Identification with Neural Networks: Understand the process of building, training, and evaluating an image classification model using neural networks.
3. Understand and Use SHAP Values: Learn what SHAP (SHapley Additive exPlanations) values are and how to use them for model interpretability.

### Dataset

For this project I've used the [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) from Kaggle.

The dataset contains images of pepper, tomato and potato plants categorized into different classes, including healthy plants and various disease conditions.

The dataset was truncated to 5 classes for simplicity (2 classes for pepper plants and 3 for potato plants), and split into training, validation, and test sets to ensure the model's robustness and generalizability.

## Model Architecture and Parameters
### Model Architecture
- **Input Layer**: The model takes an input shape of (128, 128, 3), which corresponds to images of size 128x128 pixels with three color channels (RGB). This input layer does not modify the data but passes it to the next layer.
- **Flatten Layer**: The Flatten layer converts the multi-dimensional input data into a one-dimensional array, making it suitable for the dense layers that follow.
- **Dense Layers**: Following the flatten layer, there are 4 dense layers with ReLU activation. Each dense layer has less neurons than the layer before it.
- **Output Layer**: The output layer uses the softmax activation function, and produces a probability distribution over the 5 classes for classification.

### Parameters

- Learning Rate: The model uses the Adam optimizer with a learning rate specified by the LEARNING_RATE variable.

- Number of Epochs: The model is trained for a number of epochs specified by the EPOCHS variable. Each epoch represents one complete pass through the training dataset.

- Optimizer: The model uses the Adam optimizer, which adapts the learning rate during training for improved convergence.

- Loss Function: The model is compiled with the Categorical Crossentropy loss function, which is suitable for multi-class classification problems.

- Metrics: The model tracks accuracy as a metric to evaluate its performance during training and validation.

### Callbacks

- Model Checkpoint: The ModelCheckpoint callback saves the best version of the model based on validation loss, preventing overfitting by preserving the best weights.

- Early Stopping: The EarlyStopping callback monitors the validation loss and stops training if it does not improve for a specified number of epochs (patience), as defined by the EARLY_STOPPING_PATIENCE variable.

# Discussion and Conclusions

## Discussion

### Confusion Matrix

The model demonstrates robust performance with an overall accuracy of 88%. It excels particularly in identifying **Potato___Early_blight** and has high precision in classifying healthy pepper and potato plants.

The primary area for improvement is reducing misclassifications between **Pepper__bell___Bacterial_spot**, **Pepper__bell___healthy**, and **Potato___Late_blight**. Despite these misclassifications, the model is highly effective for the task of multi-class classification of plant diseases.

### Classification Report

**Precision** - measures how many of the positive predictions made by the model are actually correct.

**Recall** - measures how well the model identifies all relevant instances.

**F1** - A measure that balances precision and recall. It's the harmonic mean of precision and recall.

The model was precise in predicting healthy pepper and potato plants, but was struggling to identify all instances of them.

It was predicting more false positives of unhealthy plants, but more or less it was great at finding most instances of plants with diseases.

```
                                precision    recall  f1-score   support

Pepper__bell___Bacterial_spot       0.81      0.83      0.82       275
       Pepper__bell___healthy       0.99      0.79      0.88       363
        Potato___Early_blight       0.92      1.00      0.96       282
         Potato___Late_blight       0.77      0.94      0.85       276
             Potato___healthy       0.97      0.85      0.90       164

                     accuracy                           0.88      1360
                    macro avg       0.89      0.88      0.88      1360
                 weighted avg       0.89      0.88      0.88      1360
```

### SHAP

In this project I've used for the first time SHAP (SHapley Additive exPlanations), which are used to explain the output of any machine learning model.

Although SHAP can provide good insight into a model, it didn't help me that much it this project to understand how the model works and why it returns the output it does. But I will use SHAP in the future projects to understand the model outputs.

### Limitations

It is hard to identify similar looking plants, and even harder to differentiate between diseases, as one plant can have multiple similar looking diseases.

But even with those limitations, the model has a close to 90% accuracy with similar looking plants and diseases.

### Future Work

Future work for this project could be:

*   Expanding the dataset (More plant species, diseases).
*   Improving the model architecture.
*   Deploying the model in real-world application.

## Conclusions

This project successfully demonstrated the application of deep learning techniques for identifying diseases in pepper and potato plants using image classification.

The neural network model achieved a high level of accuracy, effectively distinguishing between healthy and diseased plants.

These contributions highlight the potential of AI-driven solutions in early disease detection and effective crop management, paving the way for improved agricultural productivity and reduced crop losses.

![Untitled](https://github.com/nikas-belogolov/plant-disease-identification-ML/assets/30692665/06453703-9423-46f2-a332-e3a500620264)

![Untitled](https://github.com/nikas-belogolov/plant-disease-identification-ML/assets/30692665/47daac18-c272-4d4c-8676-2c67b0be0446)

![Untitled](https://github.com/nikas-belogolov/plant-disease-identification-ML/assets/30692665/486a9d92-21a6-48c1-b845-403dc8bfaa4f)

![Untitled](https://github.com/nikas-belogolov/plant-disease-identification-ML/assets/30692665/61d2ee89-078e-4545-8dde-9b4c933d5a9e)

![Untitled](https://github.com/nikas-belogolov/plant-disease-identification-ML/assets/30692665/4e021b06-ff9e-4219-a8ae-7e03a888e984)
