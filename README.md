# SpecRNet: Audio Deepfake Detection for Filipino and Tagalog Dialects

This repository is a fork of the [deepfake-whisper-features](https://github.com/piotrkawa/deepfake-whisper-features.git) repository created by Piotr Kawa. It adapts the `deepfake-whisper` framework to detect audio deepfakes specifically tailored for Filipino and Tagalog languages and dialects.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Overview
SpecRNet is an adaptation of the `deepfake-whisper` model designed to detect audio deepfakes by extracting and analyzing unique features in the audio. This project is optimized for Filipino and Tagalog speakers, making it relevant for applications in digital security, authenticity verification, and media literacy in the Philippines.

## Features
- **Language-Specific**: Trained on audio data from Filipino and Tagalog speakers.
- **Efficient Model**: Optimized for lightweight deployment, making it accessible for resource-constrained environments.
- **High Accuracy**: Delivers reliable results for detecting deepfakes in audio with Filipino/Tagalog accents.

## Installation
To get started, follow these steps to set up the environment:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/gnr2/SpecRNet.git
   cd SpecRNet
   ```

## Install Dependencies: It’s recommended to use a virtual environment:

```bash
python3 -m venv env
source env/bin/activate
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```
Download the Pre-trained Model (if available) or prepare the dataset for training.

## Usage

1. **Preprocess Audio Data**
Prepare the audio data by ensuring it is formatted to a 16kHz sampling rate, segmented into 5-second clips, and free of background noise. Place your audio files in the data/ directory, with separate folders for bonafide (real) and deepfake (fake) audio samples.

2. **Training the Model**
Train the model using the processed features:

```bash
python train.py --features_dir features/ --epochs 10 --batch_size 16
```
Adjust the --epochs and --batch_size parameters as needed for your setup.

3. **Inference on New Data**
To detect deepfakes in new audio files, use:

```bash
python predict.py --audio_path path/to/audio.wav --model_path model/SpecRNet.pth
```
This will output the likelihood of the audio being a deepfake.

4. **Evaluate the Model**
Run the evaluation script to check model performance:

```bash
python evaluate.py --features_dir features/ --model_path model/SpecRNet.pth
```
This command provides metrics such as accuracy, F1-score, and Equal Error Rate (EER) for the model.

## Model Training
For those wanting to train the model from scratch, ensure you have a dataset containing both genuine and deepfake audio samples in Filipino/Tagalog. Follow these steps:

- Place your data in the data/bonafide/ and data/deepfake/ directories.
- Run the preprocessing and feature extraction scripts as shown above.
- Train the model using the extracted features.
- Ensure your dataset includes diverse audio characteristics, such as different accents, intonations, and speaking styles, for better model generalization.

## Evaluation
Evaluate model performance using the provided evaluate.py script, which reports accuracy, F1-score, and EER (Equal Error Rate). These metrics give insights into the model’s robustness in identifying deepfakes in Filipino/Tagalog audio.

**Example usage:**

```bash
python evaluate.py --features_dir features/ --model_path model/SpecRNet.pth
```
The evaluation script outputs detailed metrics that indicate how well the model performs against deepfake audio in your specific dataset.

## Acknowledgments
This project is based on the deepfake-whisper-features repository by Piotr Kawa, which provided the foundational structure for this work. Adaptations have been made to specialize it for use with Filipino and Tagalog language data.

