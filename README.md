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
