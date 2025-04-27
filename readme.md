# Deep Learning Model Training and Evaluation Report

## 1. Introduction and Project Overview

The goal of this project was to explore and experiment with deep learning architectures for a classification task. The focus was on balancing model efficiency and accuracy, starting with data preprocessing and moving through various models, including MobileNetV2 and ResNet families (ResNet-50 and ResNet-18). Insights gained through the process led to optimized models for both accuracy and generalization.

## 2. Data Preprocessing and Preparation

Data preprocessing is the first and most crucial step in any deep learning project. The dataset was carefully loaded, cleaned, and split into training, validation, and test sets. Various image transformations were applied to improve the model's performance:

- **Resizing**: All images were resized to 224x224 pixels to match the input size expected by pre-trained models.
- **Normalization**: Image pixel values were normalized using the mean and standard deviation of the ImageNet dataset.
- **Augmentation**: Techniques like random horizontal flipping and small random rotations were applied to improve generalization and prevent overfitting.

Efficient data loaders ensured fast data processing by shuffling the data for each epoch.

## 3. MobileNetV2 Model – Initial Experimentation

### 3.1 MobileNetV2 Overview

MobileNetV2 is a lightweight CNN that balances accuracy and computational efficiency, making it ideal for resource-limited environments. A pre-trained version was fine-tuned for this classification task.

### 3.2 Architecture Modifications and Training Strategy

- **Classifier Head**: Customized to match the output classes with added layers for dropout, batch normalization, and ReLU activation.
- **Loss Function**: CrossEntropyLoss was used for multi-class classification.
- **Optimizer**: Adam optimizer with a learning rate scheduler to gradually reduce the learning rate.
- **Early Stopping**: Training was stopped when validation loss plateaued.

**Results**: The initial model achieved a Top-1 accuracy of 76.78% on the test dataset after 15 epochs.

### 3.3 Early Stopping and Model Evaluation

Top-1 and Top-5 accuracy metrics were tracked, with early stopping ensuring the model did not overfit. This model demonstrated a solid generalization capability, balancing training and validation accuracy.

## 4. MobileNetV2 – Experimentational Version

### 4.1 Architectural Changes

- **Freezing Layers**: More pre-trained layers (up to layer 10) were frozen to preserve low-level features while fine-tuning deeper layers.
- **Classifier Head**: Redesigned with LeakyReLU and ReLU activations, batch normalization, and dropout for regularization.
- **Optimizer**: Switched from Adam to AdamW for better regularization via weight decay.
- **Learning Rate Scheduler**: StepLR was implemented to reduce the learning rate every 10 epochs.

**Results**: The second version improved generalization, with a test accuracy of 47.73%.

## 5. ResNet Family Experiments – ResNet-50 and ResNet-18

### 5.1 ResNet-50 Overview

ResNet-50 is a deeper model that provides better feature extraction capacity. Similar strategies were used as in MobileNetV2, but with deeper architecture.

### 5.2 ResNet-18 Overview

ResNet-18, with fewer layers, was used for faster training and lower computational costs. Despite its smaller size, it still showed strong performance.

### 5.3 Performance Comparison

- **ResNet-50**:
  - Best Validation Accuracy: 44.06%
  - Top-1 Accuracy: 43.52%
  - Top-5 Accuracy: 62.19%
- **ResNet-18**:
  - Best Validation Accuracy: 43.89%
  - Top-1 Accuracy: 42.61%
  - Top-5 Accuracy: 60.42%

## 6. Conclusion and Insights

The experiments demonstrated valuable insights:

- **MobileNetV2**: Best for efficiency, providing good performance in less resource-intensive settings.
- **ResNet-50**: Outperformed ResNet-18 in accuracy, benefiting from its deeper architecture.
- **Early Stopping**: Crucial in preventing overfitting, ensuring optimal model performance.

Both architectures provided distinct advantages, with MobileNetV2 offering efficiency and ResNet-50 providing accuracy.

---

## 7. Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib

## 8. Installation

Clone this repository:

```bash
git clone https://github.com/irehmanar/facial-recognization.git

