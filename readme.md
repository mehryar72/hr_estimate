# Time Series Regression Project

This project implements a time series forecasting solution using deep learning models. The primary goal is to predict Hr values based on historical PPG and Acc values. The project includes three different model architectures: **Model3CNNRNN**, and **Informer**.


## Table of Contents
- [Project Overview](#project-overview)
- [Models](#models)
  - [1. Model3CNNRNN](#1-model3cnn)
  - [2. Informer](#2-informer)
- [Training the Models](#training-the-models)
- [Testing the Models](#testing-the-models)
- [Setting Up the Environment](#Setting-Up-the-Environment)

## Project Overview

The project is designed to train three models on multiple sessions of captured PPG and ACC values to predict or regress heart rate. The models process all time steps of past and future data for each heart rate prediction. The training data is split with an 85:15 ratio into training and validation sets. The validation set is used for early stopping and for the learning rate scheduler, which depends on validation loss.

The configuration file holds the default settings required for running the code. If any changes are made to the configuration, training must be repeated to ensure compatibility with inference, as both stages rely on the configuration file for model loading.

During training, the data is split into smaller sections of length 500, with a 50% overlap between consecutive sections. Alternatively, the code allows for training on longer sequences or the entire sequence, but chunking the sequences yielded better results in my experiments.

The training script trains all three models, calculates accuracies within specified ranges, and saves the results as a CSV file. The best-performing model is duplicated for inference. The LSTM and GRU models are similar in architecture. I initially added a convolutional layer at the start to leverage batch normalization because I wanted to avoid manually normalizing the data. However, I ultimately included data normalization as a preprocessing step. The mean and variance are saved during preprocessing and transferred to the inference stage.

I opted for a more complex approach by using a multi-block Inception-based convolutional layer instead of a single-layer convolution, as it yielded improved results. The convolutional layer serves several purposes: feature extraction, normalization, and dimensionality expansion. Additionally, I included a skip path combining a single-layer convolution with an RNN, which further improved the results.

Model convergence appeared to be slow, so training was capped at 400 epochs. It is possible to continue training beyond this point using a learning rate scheduler, as the validation loss tends to plateau around this stage, though I did not explore different scheduler configurations. I also did not implement gradient clipping, as the current learning rate did not require it, though it might be necessary for higher learning rates.

For LSTM models, dropout improved generalization but extended training time, so I opted for a small dropout value. Increasing dropout along with training epochs could potentially enhance results further. The GRU model, on the other hand, converged faster but was prone to overfitting sooner.

Details about the custom model can be found in the model section, as it did not perform as expected. The remainder of the README explains how to train the models and perform inference on unseen data.

## Models


### 1. Model3CNNRNN
The **Model3CNNRNN** model utilizes an inception block followed by a recurrent neural network (RNN). This model is designed to capture complex patterns in the data through:
- Inception layers that apply multiple convolutional filters.
- A bidirectional RNN (LSTM or GRU) to model temporal dependencies.
- Skip connections to improve gradient flow.

### 2. Informer

The **Informer** model is based on the transformer architecture and is originally designed for long sequence forecasting. However, I adapted it for classification in this project. It incorporates:
- A data embedding layer to transform input features.
- A multi-layer encoder with attention mechanisms to focus on the most relevant parts of the input sequence.
- A linear layer for final predictions.

I chose to experiment with a transformer-based model, believing it would be well-suited for the task. I also considered trying XGBoost, but time constraints prevented me from doing so. Despite my efforts, the Informer model struggled to converge. I tested several well-known transformer variants, but none performed satisfactorily.

The primary issue appeared to be the positional encoding, which had a severely negative impact on the loss progression. When positional encoding was included, the loss increased or plateaued. However, once I removed the positional encoding, the loss began to decrease, indicating that the model was finally learning.

## Training the Models

To train the models, follow these steps:

1. **Prepare your dataset**: Ensure your time series data is in CSV format and organized in `TrainData` directory.

2. **Configure the settings**: Modify the configuration settings in the `config.py` file to specify parameters such as learning rate, batch size, and model type.

3. **Run the training script**: Execute the `main.py` script to start training. You can specify the model type (e.g., `BiLSTMRegressor`, `Model3CNNRNN`, or `Informer`) in the script.

   ```bash
   python main.py
   ```

4. **Monitor training**: The training process will log the training and validation losses, as well as accuracy metrics, to a CSV file.

## Testing the Models

To evaluate the performance of the trained models:
I must have tested not chunking during infernce for faster inference to or use batching, its slow now.
1. **Load the best model**: The `Inference.py` script loads the best model based on validation performance.

2. **Prepare the test dataset**: Ensure your test dataset is placed in `ValidData` path for`Inference.py` script.

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
