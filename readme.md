# Time Series Regression Project

This project implements a time series forecasting solution using deep learning models. The primary goal is to predict Hr values based on historical PPG and Acc values. The project includes three different model architectures: **Model3CNNRNN**, and **Informer**.


## Table of Contents
- [Project Overview](#project-overview)
- [Models](#models)
  - [1. Model3CNNRNN](#2-model3cnn)
  - [2. Informer](#3-informer)
- [Training the Models](#training-the-models)
- [Testing the Models](#testing-the-models)
- [Setting Up the Environment](#Setting Up the Environment)

## Project Overview

The project is desinged to train 3 models on multiple sessions of captured PPG,ACC values to predict or regress the heartrate. The models see all time steps of past and future for each heart rate prediction.

## Models

### 1. BiLSTMRegressor
The **BiLSTMRegressor** model combines convolutional layers with a bidirectional LSTM. It captures both local and long-range dependencies in the time series data. The architecture includes:
- Convolutional layers for feature extraction.
- A bidirectional LSTM to process the sequential data.
- Skip connections to enhance learning.

### 2. Model3CNNRNN
The **Model3CNNRNN** model utilizes an inception block followed by a recurrent neural network (RNN). This model is designed to capture complex patterns in the data through:
- Inception layers that apply multiple convolutional filters.
- A bidirectional RNN (LSTM or GRU) to model temporal dependencies.
- Skip connections to improve gradient flow.

### 3. Informer
The **Informer** model is based on the transformer architecture and is designed for long sequence forecasting. It employs:
- A data embedding layer to transform input features.
- A multi-layer encoder with attention mechanisms to focus on relevant parts of the input sequence.
- A linear layer for final predictions.

## Training the Models

To train the models, follow these steps:

1. **Prepare your dataset**: Ensure your time series data is in CSV format and organized in a directory.

2. **Configure the settings**: Modify the configuration settings in the `config.py` file to specify parameters such as learning rate, batch size, and model type.

3. **Run the training script**: Execute the `main.py` script to start training. You can specify the model type (e.g., `BiLSTMRegressor`, `Model3CNNRNN`, or `Informer`) in the script.

   ```bash
   python main.py
   ```

4. **Monitor training**: The training process will log the training and validation losses, as well as accuracy metrics, to a CSV file.

## Testing the Models

To evaluate the performance of the trained models:

1. **Load the best model**: The `Inference.py` script loads the best model based on validation performance.

2. **Prepare the test dataset**: Ensure your test dataset is in the correct format and specify the path in the `Inference.py` script.

3. **Run the inference script**: Execute the `Inference.py` script to evaluate the model on the test dataset.

   ```bash
   python Inference.py
   ```

4. **View results**: The script will print the accuracy results for different margins, indicating how well the model performed on the test data.

## Setting Up the Environment

There are two ways to recreate the environment: using Conda or `pip`.

### Using Conda
   ```bash
   conda env create -f environment.yml
   conda activate <environment_name>
   ```
### Using Pip
    ```bash
    pip install -r requirements.txt

    ```
