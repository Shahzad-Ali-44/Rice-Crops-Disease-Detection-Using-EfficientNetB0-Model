# Rice Crops Disease Detection Using EfficientNetB0 Model

## Overview
This project employs the EfficientNetB0 deep learning model to accurately detect diseases in rice crops. By leveraging transfer learning with EfficientNetB0, fine-tuned on a custom dataset, the model achieves impressive performance with reliable classification capabilities. It provides a scalable and efficient solution for early detection and management of crop diseases.

## Features
- High-accuracy detection of rice crop diseases.
- EfficientNetB0 architecture optimized for both speed and accuracy.
- Customized preprocessing and data augmentation for robust training.
- Outputs predictions with class labels and confidence scores.

## Dataset
- **Source**: A curated dataset of rice crop images, including healthy and diseased samples.
- **Classes**: The dataset includes **9 classes**, representing specific diseases and healthy crops.
- **Preprocessing**:
  - Images resized to **128x128 pixels**.
  - Normalization using EfficientNet's `preprocess_input` function.
  - Data augmentation including shearing, zooming, and horizontal flipping.

## Methodology
1. **Data Preparation**:
   - Used `ImageDataGenerator` for preprocessing and augmentation.
   - Loaded images in batches of size 64 for efficient training.

2. **Model Architecture**:
   - EfficientNetB0 pre-trained on ImageNet.
   - Added custom layers:
     - Global Average Pooling for dimensionality reduction.
     - Fully connected dense layers for classification.
     - Softmax output layer for multi-class predictions.

3. **Training**:
   - Trained for **10 epochs** with categorical cross-entropy loss and the Adam optimizer.
   - Training and validation steps calculated for efficient utilization of data.

4. **Evaluation**:
   - Achieved **98% accuracy** on the validation set.
   - Evaluated with confusion matrix and performance metrics visualization.

## Dependencies
- Python
- TensorFlow/Keras
- NumPy
- Matplotlib
- scikit-learn

## Results
- **Accuracy**: 98%
- **Evaluation Metrics**: High precision, recall, and F1-score for all classes.
- Confusion matrix and prediction visualizations confirm the model's robustness.

## Visualizations
- Confusion Matrix: Displays class-wise performance.
- Accuracy and Loss Graphs: Line and bar plots showcasing training and validation trends.
- Predictions: Grid of sample images with actual and predicted labels, alongside confidence scores.

## Future Improvements
- Expand the dataset with additional disease classes and varied conditions.
- Deploy the model for real-time detection via mobile or web platforms.
- Integrate IoT solutions for real-time monitoring and disease alerts.

## Contributing
Contributions are welcome! Feel free to fork the repository, make improvements, and submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Dataset providers for their contributions.
- TensorFlow and Keras for enabling efficient model development.
- The open-source community for providing invaluable tools and resources.
