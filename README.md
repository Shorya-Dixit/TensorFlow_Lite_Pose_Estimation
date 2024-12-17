# Pose Estimation Model

## Project Overview
This project focuses on optimizing a TensorFlow Lite (TFLite) pose estimation model for mobile inference. It includes model quantization, inference on real images, and visualization of detected body keypoints.

The keypoints are mapped to body parts based on the COCO dataset format. This optimized model is lightweight and suitable for mobile devices with fast inference times.

---

## Key Features
- **TensorFlow Lite Model**: Pose estimation optimized for mobile inference.
- **Model Quantization**: Reducing model size for performance efficiency.
- **Inference Visualization**: Visualizing keypoints on input images.
- **Inference Time Measurement**: Measuring and reporting inference time in milliseconds.
- **Keypoint Mapping**: Mapping keypoints to body parts.
- **Google Colab Environment**: No setup required to run the code directly in Colab.

## Project Directory
This directory structure is for Colab users to run the code interactively.

├── real_image.jpg # Input test image ├── model.tflite # TensorFlow Lite optimized model ├── keypoint_mapping.py # Script to print the keypoint mapping table ├── inference_visualization.py # Inference script to visualize keypoints └── README.md # Project documentation
