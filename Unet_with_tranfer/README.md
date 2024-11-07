
# MRI Image Segmentation with U-Net

This project implements a U-Net model designed for segmentation of medical images, specifically 3D MRI scans. The model takes MRI images as input and leverages a pre-trained VGG16 network as the encoder. The decoder uses several up-sampling layers to reconstruct the image to its original dimensions.

## Libraries
The code utilizes several key libraries for machine learning and image processing:
- **numpy**: For mathematical operations
- **nibabel**: To load MRI images in NIfTI format (`.nii` files)
- **PIL** (Pillow): For image processing and conversion
- **sklearn**: For data preprocessing and train-test splitting
- **tensorflow** and **keras**: To build and train the U-Net model

## Code Overview

### 1. File Parsing and Preprocessing
- `split_filename`: A helper function that extracts the file path, base name, and file extension, handling compressed files (`.gz`).
- `ProcessImage`: Processes 3D MRI images in NIfTI format, extracting each 2D slice along the z-axis. If `mode='mask'`, treats the images as segmentation masks; otherwise, normalizes the slices to a range between 0 and 1.
- `LoadData`: Loads both raw MRI images and their corresponding segmentation masks from the specified directory.

### 2. Label Encoding and Class Weights Calculation
- `Encode_label`: Reshapes and encodes the segmentation masks using `LabelEncoder` to ensure numeric labels. Computes class weights to address class imbalance, improving model performance.

### 3. Data Splitting
- The data is split into training and testing sets using `train_test_split` from `sklearn`. Categorical encoding is applied to the target masks for multi-class segmentation.

### 4. Model Definition
- **VGG16 Initialization**: A pre-trained VGG16 model (without the top classification layers) is modified to be compatible with grayscale images of shape (512, 512, 1).
- **Custom U-Net Architecture**:
  - `upsample`: Defines the up-sampling (deconvolution) layers used in the U-Net’s decoder path.
  - `buildUNet`: Constructs the U-Net architecture by combining the modified VGG16 model as the encoder and custom up-sampling layers as the decoder. Skip connections are added to preserve spatial information, crucial for image segmentation.

### 5. Training the Model
- The `U-Net` model is compiled using the Adam optimizer and categorical cross-entropy loss, which is suitable for multi-class segmentation.
- `model.fit` trains the model on the training data, tracking accuracy and loss on the validation data for three epochs.

## Summary
This project provides an end-to-end pipeline for loading, processing, and training a deep learning model for MRI image segmentation. The U-Net model’s encoder benefits from transfer learning via pre-trained VGG16 layers, while the decoder is designed to perform up-sampling and segmentation on the processed images.

---

**Note**: This code is tailored for use with medical images in NIfTI format. Ensure data is structured according to the specified directory path before running.
