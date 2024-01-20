import numpy as np


def bias_init(b):
    return np.zeros_like(b)


def bias_init_ones(b):
    return np.ones_like(b)


def bias_init_zeros(b):
    return np.zeros_like(b)


def weight_init_randn(w):
    return np.random.randn(*w.shape)


def weight_init_randn(w):
    return np.random.randn(*w.shape)


def weight_init_normal(w):
    return np.random.normal(0, 1.0, w.shape)


def weight_init_LeCun(w):
    # LeCun's
    # W shape (out_features, in_features)
    stdv = 1.0 / np.sqrt(w.shape[1])
    return np.random.uniform(-stdv, stdv, w.shape)


def weight_init_LeCun_CNN(w):
    # LeCun's
    # W shape (out_channel, in_channel, kernel_size, kernel_size) or
    # W shape (out_channel, in_channel, kernel_size)
    fan_in = np.prod(w.shape[1:])
    stdv = 1.0 / np.sqrt(fan_in)
    return np.random.uniform(-stdv, stdv, w.shape)


def weight_init_He(w):
    # Kaiming He's initialization
    # W shape (out_features, in_features)
    stdv = np.sqrt(2.0 / w.shape[1])
    return np.random.normal(0, stdv, w.shape)


def weight_init_He_CNN(w):
    # Kaiming He's initialization
    # w shape (out_channels, in_channels, kernel_size) for 1D CNN
    # w shape (out_channels, in_channels, kernel_size, kernel_size) for 2D CNN
    fan_in = np.prod(w.shape[1:])
    stdv = np.sqrt(2.0 / fan_in)
    return np.random.normal(0, stdv, w.shape)
