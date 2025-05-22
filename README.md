# Computer-Vision-for-Smoker-Detection-Enhancing-Public-Health-Surveillance using VLGG16

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

### VGG16 Model
The VGG16 model is a convolutional neural network architecture that was introduced by the Visual Geometry Group (VGG) at the University of Oxford. It is known for its simplicity and effectiveness in image classification tasks. The model consists of 16 layers, including 13 convolutional layers and 3 fully connected layers. The VGG16 model has been pre-trained on the ImageNet dataset, which contains over 1 million images across 1000 categories. This pre-training allows the model to learn general features that can be fine-tuned for specific tasks, such as smoker detection.

### Results
The VGG16 model achieved an accuracy of 96.43% on the test set. The precision, recall, and F1-score were 0.96, 0.97, and 0.96 respectively. The confusion matrix and classification report are as follows:
 
![Screenshot 2025-05-22 093347](https://github.com/user-attachments/assets/475b8ea8-11fb-4a94-b317-6bf9b370e303)



Classification Report:
              precision    recall  f1-score   support

           0       0.95      0.92      0.93        60
           1       0.92      0.95      0.93        60

    accuracy                           0.93       120
   macro avg       0.93      0.93      0.93       120
weighted avg       0.93      0.93      0.93       120
