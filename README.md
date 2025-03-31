# Shadow Removal Project

This project implements a shadow removal system using deep learning with a Flask-based web interface.

## Features

- Shadow removal from images using a GAN-based model
- Video processing (frame-by-frame shadow removal)
- Interactive web interface with Bootstrap
- Side-by-side comparison of original and processed images
- Download functionality for processed results

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download the SRD dataset and place it in `data/SRD/`
4. Train the model: `python train.py`
5. Run the Flask application: `python app.py`

## Dataset

The model is trained on the SRD (Shadow Removal Dataset) which contains pairs of shadow and shadow-free images. The dataset should be placed in the following structure:

data/SRD/
train/
shadow/
shadow_free/
test/
shadow/
shadow_free/

## Usage

1. Access the web interface at `http://localhost:5000`
2. Upload an image
3. View the processed result with side-by-side comparison
4. Download the processed file if desired

## Model Architecture

The system uses a Generative Adversarial Network (GAN) with:
- Generator: U-Net architecture for image-to-image translation
- Discriminator: PatchGAN classifier
- Loss function: Combination of L1 loss and adversarial loss

## Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- Flask
- Bootstrap 4