# Deep Learning (NumPy implementation)

This project implements a modular deep learning framework from scratch using NumPy, 
including neural network architectures, training pipelines, and evaluation tools.

The goal is to expose the full modeling pipeline—forward/backward propagation, 
optimization, and evaluation—without relying on automatic differentiation libraries.

## Motivation
Modern frameworks such as PyTorch abstract away implementation details. 
This project reconstructs these components explicitly to:

- Understand gradient-based learning at a systems level
- Enable full control over model behavior and debugging
- Provide a foundation for extending and experimenting with new architectures

## Tech Stack
Python
NumPy

## Features
- Neural network architectures:
  - Multi-layer perceptrons (MLP)
  - 1D and 2D convolutional neural networks (CNN)
  - Recurrent neural networks (RNN)
- Manual backpropagation implementation
- Modular components:
  - Layers (`nn/modules`)
  - Optimizers (`optim`)
  - Evaluation tools (`evaluation`)
- End-to-end training pipeline


Folder "project" structure:
- evaluation/: contain modules for assessing models
- examples/: contain actual applications to demonstrate the models
- models/: contain the main scripts for the models
- nn/modules/: contain modules ingredients for the models
- optim/: contain algorithms for optimization
- utils/: contain utility functions for obtaining and processing data

Check out the demonstration examples here: https://github.com/weili-code/DeepLearning_numpy/tree/main/project/examples

Note: 
The implementations are benchmarked against PyTorch models to verify correctness
and numerical behavior. This project prioritizes clarity and transparency over computational efficiency. It is designed for experimentation, validation, and extension of deep learning models.


A PyTorch-based implementation of similar models is available here:
https://github.com/weili-code/DeepLearning_pytorch. 

## Acknowledgments

This project draws on concepts and methodologies commonly taught in leading 
machine learning and deep learning courses such as:

- CMU 11-785 (Bhiksha Raj & Rita Singh)  
- CS229 (Stanford, Andrew Ng et al.)  
- STAT 453 (Sebastian Raschka)


## References: 
- Deep Learning, 2016, Ian Goodfellow et al, https://www.deeplearningbook.org/
- Understanding Deep Learning 2023, Simon J.D. Prince, https://udlbook.github.io/udlbook/
- Probabilistic Machine Learning 2022, Kevin P. Murphy, https://probml.github.io/pml-book/book1.html
- Probabilistic Machine Learning 2023 (advanced), Kevin P. Murphy, https://probml.github.io/pml-book/book2.html
