from collections import OrderedDict

import numpy as np
from nn.modules.parameters import Parameter


class Layer():
    """Base class for all neural network layers such as linear, batchnorm
    """

    def __init__(self, *args, **kwargs):
        self._layers = OrderedDict()
        self._parameters = OrderedDict()
        self.train_mode = True

    def train(self, mode=True):
        """Recursively sets the traiing mode to `mode` for all submodules.
        """
        self.training_mode = mode
        for layer in self.children():
            layer.train(mode)

    def eval(self):
        """Recursively sets the training mode to evaluation for all submodules.
        """
        self.train(mode=False)

    def forward(self, input):
        raise NotImplementedError()

    def backward(self, delta_in):
        raise NotImplementedError()