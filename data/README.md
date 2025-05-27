# Dataset Information for Rat Detection System

This README file provides essential information about the dataset used in the Rat Detection System project.

## Dataset Overview

The dataset consists of images used to train a Convolutional Neural Network (CNN) for detecting rats in various environments. The images are labeled as either containing a rat or not containing a rat.

## Accessing the Data

The dataset can be accessed from [insert dataset source or link here]. Ensure that you have the necessary permissions to download and use the data.

## Data Structure

The dataset is organized into the following structure:

- **train/**: Contains training images.
- **validation/**: Contains validation images.
- **test/**: Contains test images.

Each folder contains subfolders for each class (e.g., `rat` and `no_rat`).

## Preprocessing Steps

Before using the dataset for training the CNN, the following preprocessing steps should be performed:

1. **Image Resizing**: Resize images to a consistent size (e.g., 128x128 pixels).
2. **Normalization**: Scale pixel values to the range [0, 1].
3. **Data Augmentation**: Apply techniques such as rotation, flipping, and zooming to increase dataset diversity.

## Usage

To use the dataset in your project, follow these steps:

1. Download the dataset and extract it to the appropriate directory.
2. Update the paths in the `notebooks/rat_detection_cnn.ipynb` file to point to the dataset location.
3. Run the notebook to load, preprocess, and train the CNN model.

## Acknowledgments

We acknowledge the sources from which the dataset was obtained and any contributors to the dataset.