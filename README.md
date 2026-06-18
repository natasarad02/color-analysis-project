
# Automatic Seasonal Color Palette Classification from Face Images

This project implements an image-classification system for automatic seasonal color analysis. Based on a face image, the system predicts one of four seasonal color types: **Spring**, **Summer**, **Autumn**, or **Winter**.

Seasonal color analysis is traditionally performed manually by experts, which makes it subjective, time-consuming, and less accessible. The goal of this project is to explore whether convolutional neural networks can learn subtle visual differences in skin tone, contrast, and color saturation from face images.

## Project Overview

The project uses a manually created dataset of **800 annotated face images**, divided equally into four seasonal classes:

- Spring
- Summer
- Autumn
- Winter

Each class contains **200 images**. The dataset is split into training, validation, and test subsets using a **70:15:15** ratio.

| Dataset split | Images per class | Total images |
|---|---:|---:|
| Training | 140 | 560 |
| Validation | 30 | 120 |
| Test | 30 | 120 |

The main classification models compared in this project are:

- **MobileNetV3**
- **ResNet50**

Both models were initialized with ImageNet pretrained weights and then fine-tuned on the custom seasonal color dataset.

## Main Features

- Manual dataset creation and annotation for seasonal color classification
- Face extraction and preprocessing using **MediaPipe Face Mesh**
- Image resizing to **224 × 224 px**
- HSV analysis of extracted facial regions
- Transfer learning with pretrained CNN architectures
- Comparison of MobileNetV3 and ResNet50
- Evaluation using accuracy, precision, F1-score, confusion matrix, and visual error analysis

## Methodology

### 1. Dataset Preparation

Since no public dataset with predefined seasonal color labels was available, a custom dataset was created manually. Images were collected from free image sources such as Unsplash, Pexels, and CelebA, then manually assigned to one of the four seasonal categories.

Each portrait was cropped so that the face was the main focus. Images were then resized to **224 × 224 pixels**.

### 2. Face Extraction

The project uses **MediaPipe Face Mesh** to detect facial landmarks. For each image, 468 face landmarks are detected and used to crop the facial region. The detected bounding box is expanded by 20% to include important visual features such as hair and jawline.

This preprocessing step reduces the influence of background, clothing, and other irrelevant image elements.

### 3. HSV Color Analysis

To better understand the dataset, HSV color components were analyzed on extracted face regions:

- **Hue**: related to warm/cool color tone
- **Saturation**: related to muted or vivid coloring
- **Value**: related to brightness and contrast

The analysis showed that seasonal categories are not separated by strict thresholds, but by subtle distribution shifts. Saturation was the most useful component for distinguishing between seasons.

### 4. Model Training

The project compares two pretrained CNN architectures:

- **MobileNetV3**, selected because of its efficiency and strong performance on face-related image classification tasks
- **ResNet50**, used as a deeper reference architecture for comparison

Data augmentation was applied to reduce overfitting:

- Random rotation up to 20 degrees
- Horizontal flipping
- Mild brightness changes

The most stable hyperparameter configuration was:

```text
Learning rate: 0.0001
Batch size: 8
Optimizer: AdamW
Weight decay: 0.0001
Epochs: 7
```

## Results

| Model | Accuracy | Precision | F1-score |
|---|---:|---:|---:|
| MobileNetV3 | 61.57% | 0.64 | 0.62 |
| ResNet50 | 48.33% | 0.4966 | 0.4886 |

MobileNetV3 achieved better generalization on unseen test data. ResNet50 showed stronger signs of overfitting and performed worse on the test set despite similar validation accuracy.

## Error Analysis

The most common classification errors occurred between visually similar seasonal types:

- Autumn and Spring
- Summer and Spring

These mistakes are expected because the differences between these classes are subtle. Spring and Autumn both have warmer undertones, while Summer and Spring may differ mainly in saturation and softness.

The main reasons for classification errors include:

1. **Low image resolution** after resizing to 224 × 224 px
2. **Subjective manual annotation** of seasonal types
3. **Borderline examples** where even human classification is uncertain
4. **Visual similarity between classes**
5. Variations in lighting, image quality, pose, and exposure

## Observed Seasonal Characteristics

| Season | Observed characteristics |
|---|---|
| Spring | Lighter complexion, warm undertone, visible blush |
| Summer | Pale complexion, cooler undertone, muted colors |
| Autumn | Warm yellowish complexion, darker tones |
| Winter | Darkest complexion, cool undertone, high contrast |

## Conclusion

The results show that CNN-based transfer learning can be applied to automatic seasonal color classification from face images. MobileNetV3 performed better than ResNet50 and proved to be more suitable for this task on a small manually annotated dataset.

However, the task remains challenging because seasonal color types are fine-grained and subjective. Higher accuracy would likely require a larger, more consistently annotated dataset collected under more controlled lighting conditions.

## Future Work

Possible improvements include:

- Expanding the dataset with more images
- Improving annotation consistency by using multiple annotators
- Testing additional CNN or vision transformer architectures
- Developing a domain-specific model for seasonal color classification
- Building a client-server or web application where users can upload a face image and receive a predicted seasonal type
- Adding explainability features that describe which visual characteristics influenced the prediction

## Technologies Used

- Python
- TensorFlow / Keras or PyTorch
- MediaPipe Face Mesh
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- MobileNetV3
- ResNet50

## Authors

- Nataša Radmilović
- Katarina Petrović

Faculty of Technical Sciences, University of Novi Sad

## References

1. H. Kim, S. Lee, and J. Park, *An Automatic Virtual Makeup Scheme Based on Personal Color Analysis*, Journal of the Korea Society of Computer and Information, 2017.
2. D. Bătrânu et al., *Recognizing Human Races through Machine Learning—A Multi-Network, Multi-Features Study*, Mathematics, 2021.
3. M. H. Shuvon et al., *Skin Disease Detection and Classification Using Deep Learning*, Brac University, 2022.
4. C. Lugaresi et al., *MediaPipe: A Framework for Building Perception Pipelines*, arXiv, 2019.
5. A. Howard et al., *Searching for MobileNetV3*, ICCV, 2019.
6. K. He et al., *Deep Residual Learning for Image Recognition*, CVPR, 2016.
