# Image Classification

## Purpose

This is my final project for the Break Through Tech Machine Learning Fundamentals Course. I chose to work with the CIFAR-10 dataset to deepen my understanding of Convolutional Neural Networks (CNNs) and gain hands-on experience applying machine learning to real-world image classification tasks. The project involved thorough data preprocessing, architecture experimentation, and hyperparameter optimization to enhance model performance. Through this, I aimed to build a solid foundation in deep learning workflows and improve my practical coding skills using TensorFlow.

## Contents

- `DefineAndSolveMLProblem.ipynb`: Jupyter notebook that defines the full image classification pipeline, including data preprocessing, CNN architecture, training, and evaluation using TensorFlow.
- `Image_Classification_Model.keras`: Saved TensorFlow model trained on the CIFAR-10 dataset, serialized in Keras format.

## Datasets 

The project uses the [CIFAR-10 dataset](https://keras.io/api/datasets/cifar10/) provided by Keras.

## Model Architecture & Approach

- The pixel values from the CIFAR-10 dataset are used directly as features; no additional feature engineering is needed.
- Input data is normalized by scaling pixel values to a [0, 1] range, which improves convergence during training.
- A Convolutional Neural Network (CNN) is used for classification.
- The architecture consists of 4 convolutional layers, each followed by batch normalization and ReLU activation functions.
- A final dense layer with softmax activation is used to output class probabilities for this multi-class classification task.
- The model is compiled using:
  - **Loss Function**: Sparse Categorical Crossentropy (due to integer labels),
  - **Optimizer**: Stochastic Gradient Descent (SGD),
  - **Metric**: Accuracy.
- The model is trained and evaluated using the test set.
- To improve generalization and performance:
  - Grid search is used for hyperparameter optimization.
  - Dropout and pooling layers are added to mitigate overfitting and improve accuracy.

## Results

| Metric   | Training Set | Testing Set |
|----------|--------------|-------------|
| Loss     | 0.5263       | 0.5650      |
| Accuracy | 81.55%       | 80.46%      |

The model shows consistent performance between training and testing, indicating good generalization without significant overfitting.


## Deployment

Download the `.keras` file and load it in a python file to use.

```python
import tensorflow as tf
model = tf.keras.models.load_model("Image_Classification_Model.keras")
