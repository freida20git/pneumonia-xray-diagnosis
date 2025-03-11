# Pneumonia X-ray Diagnosis

This project aims to build a deep learning model for detecting pneumonia from chest X-ray images using PyTorch. The model is served using a Flask web application.

## Overview
The project is divided into the following parts:
- **Model Training**: Training a convolutional neural network (CNN) using labeled X-ray images to classify images as 'Normal' or 'Pneumonia'.
- **Flask API**: Providing an API for model inference that accepts images and returns predictions.
- **Frontend Interface**: (To be developed) A simple user interface for uploading images and viewing predictions.

## Data Used
The dataset used for training and evaluation is the [Chest X-ray Images (Pneumonia) dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) from Kaggle. It contains 5,863 X-ray images with labels:
- **Normal**: 1,583 images
- **Pneumonia**: 4,273 images

The dataset is divided into training, validation, and test sets.

## Installation
To run this project, clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

## Model Architecture
The model is a simple convolutional neural network (CNN) built with PyTorch. It consists of convolutional layers followed by ReLU activation and max-pooling, with fully connected layers for classification.

## Usage
### 1. Training the Model
You can train the model by running the `X_ray_DL_model.ipynb` notebook.

### 2. Running the Flask App
To run the Flask app:
```bash
python flask_app.py
```
The API will be available at `http://127.0.0.1:5000/predict`.

### 3. Making Predictions
Send a POST request to the API with an image file. Example using `curl`:
```bash
curl -X POST -F "file=@path_to_image.jpg" http://127.0.0.1:5000/predict
```

## Model File
The model file `trained_model_20epochs_1.pth` can be downloaded from the following link:
[Download Model from Google Drive](https://drive.google.com/file/d/1tCPDSut7EN-MtlzI-dtfTIbHvYJYUSbk/view?usp=drive_link)

## File Structure
- `X_ray_DL_model.ipynb`: Model training notebook
- `flask_app.py`: Flask application file
- `model_loader.py`: Model loading utility
- `trained_model_20epochs_1.pth`: Trained PyTorch model file
- `requirements.txt`: List of dependencies
- `README.md`: Project documentation.

