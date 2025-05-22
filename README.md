# Computer-Vision-for-Smoker-Detection-Enhancing-Public-Health-Surveillance using VGG16 and ResNet-50

This project endeavours to make significant contributions to the field of smoker detection and public health surveillance. The outcomes of this research have the potential to inform policy decisions, guide smoking cessation interventions, and contribute to the creation of healthier environments worldwide.

The dataset is sourced from Kaggle which contains the images of individuals smoking and non-smoking. The dataset contains 1120 images in total, that is divided equally in two classes, in which 560 images belongs to smokers and remaining 560 images belongs to non-smokers. All the images in the dataset are resized to a resolution of 250*250.

Dataset Link: [Kaggle](https://www.kaggle.com/datasets/sujaykapadnis/smoking)

## Data Augmentation
It is a technique used to increase the diversity of training data without collecting new data. This is particularly useful in image classification tasks to improve the model’s robustness and generalizability by enlarging the training dataset. Augmentation techniques involve applying various transformations to the original images, such as rotations, transitions, flips, and changes in brightness and contrast. These transformations help the model become invariant to these changes, making it more reliable when encountering variations in real-world data.

### Dataset
The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/sujaykapadnis/smoking) and contains:
- 1120 total images (560 smoking / 560 non-smoking)
- 250×250 pixel resolution
- Balanced classes with equal distribution
- Pre-processed and resized images

## VGG16 Model
The VGG16 model is a convolutional neural network architecture that was introduced by the Visual Geometry Group (VGG) at the University of Oxford. It is known for its simplicity and effectiveness in image classification tasks. The model consists of 16 layers, including 13 convolutional layers and 3 fully connected layers. The VGG16 model has been pre-trained on the ImageNet dataset, which contains over 1 million images across 1000 categories. This pre-training allows the model to learn general features that can be fine-tuned for specific tasks, such as smoker detection.

### Results
The VGG16 model achieved an accuracy of 90.0% on the test set. The precision, recall, and F1-score were 0.90, 0.90, and 0.90 respectively. The confusion matrix and classification report are as follows:

![Screenshot 2025-05-22 093347](https://github.com/user-attachments/assets/475b8ea8-11fb-4a94-b317-6bf9b370e303)

Classification Report:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.95      | 0.92   | 0.93     | 60      |
| 1     | 0.92      | 0.95   | 0.93     | 60      |

Accuracy: 0.93
Macro Avg: 0.93
Weighted Avg: 0.93

## ResNet-50 Model
The ResNet-50 model is a convolutional neural network architecture that was introduced by Microsoft Research. It is known for its depth and the use of residual connections, which help in training very deep networks. The model consists of 50 layers, including 49 convolutional layers and 1 fully connected layer. The ResNet-50 model has been pre-trained on the ImageNet dataset, which contains over 1 million images across 1000 categories. This pre-training allows the model to learn general features that can be fine-tuned for specific tasks, such as smoker detection.

### Results
The ResNet-50 model achieved an accuracy of 92.5% on the test set. The precision, recall, and F1-score were 0.93, 0.92, and 0.92 respectively. The confusion matrix and classification report are as follows:

![Confusion Matrix for ResNet-50](path/to/resnet50_confusion_matrix.png)

Classification Report:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.93      | 0.92   | 0.92     | 60      |
| 1     | 0.92      | 0.93   | 0.92     | 60      |

Accuracy: 0.925
Macro Avg: 0.925
Weighted Avg: 0.925

## Comparison of Models
Comparing the two models:
- **ResNet-50**:
  - Accuracy: 92.5%
  - Precision: 0.93
  - Recall: 0.92
  - F1-score: 0.92
  - AUC: 92.5%
- **VGG16**:
  - Accuracy: 90.0%
  - Precision: 0.90
  - Recall: 0.90
  - F1-score: 0.90
  - AUC: 90.0%

**Conclusion**: The ResNet-50 model outperforms the VGG16 model in all metrics, making it the better choice for this task.
