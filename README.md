# Text Classification using Neural Network 
Classify sentiment in [IMDB](https://huggingface.co/datasets/imdb) dataset using custom built recurrent neural network.  
Demo: https://huggingface.co/spaces/sadjava/sentiment-analysis

## Features
⚡ Text Classification  
⚡ Cutsom RNN  
⚡ Attention Block  


## Table of Contents
- [Objective](#objective)
- [Dataset](#dataset)
- [Evaluation Criteria](#evaluation-criteria)
- [Solution Approach](#solution-approach)

## Objective
I'll build a neural network using PyTorch. The goal here is to build a system to identify the sentiment on a given movie review.

## Dataset
- Dataset consists of 40,000 training reviews and 10,000 validation reviews.
- Every image in the dataset will belong to one of the two classes.

| Label	| Description |
|--- | ---|
|0|	Negative|
|1|	Positive|


## Evaluation Criteria

### Loss Function  
Binary Cross Entropy Loss is used as the loss function during model training and validation 

### Performance Metric
`accuracy` is used as the model's performance metric on the test-set 

<img src="https://github.com/sssingh/fashion-mnist-classification/blob/master/assets/accuracy.png?raw=true">

`F1-score`, `precision` and `recall` are also computed.

## Solution Approach
- Training dataset along with validation dataset are downloaded from [hugging face](https://huggingface.co/datasets/imdb).
- Training dataset is then further split into training (40,000 samples) and validation (10,000) samples sets
- The training and datasets are then wrapped in PyTorch `DataLoader` object with collate function using `pad_sequence` and `pack_padded_sequence`, so that we can iterate through them with ease. A `batch_size` can be configured.
