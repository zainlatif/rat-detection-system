# Rat Detection System

This project implements a rat detection system using Convolutional Neural Networks (CNN). The goal is to accurately identify and classify images of rats.

## Project Structure

- **data/**: Contains the dataset used for training and testing the model.
  - **README.md**: Information about the dataset, including access and preprocessing details.
  
- **notebooks/**: Jupyter notebooks for experimentation and model training.
  - **rat_detection_cnn.ipynb**: Implementation of the CNN for rat detection, including data loading, model training, evaluation, and visualization of results.
  
- **src/**: Source code for the project.
  - **cnn_model.py**: Defines the CNN architecture, including model class, training, and evaluation methods.
  - **utils.py**: Contains utility functions for data preprocessing, image augmentation, and other helper functions.
  
- **requirements.txt**: Lists the Python dependencies required for the project, including TensorFlow, Keras, NumPy, and others.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd rat-detection-system
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare the dataset as described in `data/README.md`.

## Usage

- Use the Jupyter notebook in the `notebooks/` directory to train and evaluate the CNN model.
- Refer to the `src/` directory for the underlying code and architecture of the model.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.