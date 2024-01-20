import numpy as np


class Activation(object):
    """
    Interface for activation functions f
    Z: pre-activation
    A: output
    A_current=f_prev(Z_prev)

    Allowing elementwise and vector activation (e.g, softmax).

    Attrs:
        A: value of activation
    """

    # Note that in this super class, the activation functions are scalar operations.
    # i.e, they shouldn't change the shape of the input.

    def __init__(self):
        self.A = None

    def __call__(self, Z):
        return self.forward(Z)

    def forward(self, Z):
        raise NotImplemented

    def backward(self):
        raise NotImplemented


class Identity(Activation):
    """
    Identity function (elementwise)

    Attrs:
        A: value of activation

    Methods:
        fordward
        backward
    """

    def forward(self, Z):
        """
        Args:
            Z: output from previous layer

        Returns:
            self.A: =Z
        """

        self.A = Z

        return self.A

    def backward(self):
        dAdZ = np.ones(self.A.shape, dtype="f")

        return dAdZ


class Sigmoid(Activation):
    """
    Sigmoid function (elementwise)
    Can be used in intermediate layers or output layer.

    Attrs:
        A: value of activation

    Methods:
        fordward
        backward
    """

    def forward(self, Z):
        """
        Z: np.ndarray (batch size, num_features)
        For output layer with binary classification, use num_features=1.
        For use in intermediate layers,
        Z is allowed to be multidimensional.
        """
        self.A = 1 / (1 + np.exp(-Z))

        return self.A

    def backward(self):
        dAdZ = self.A * (1 - self.A)

        return dAdZ


class Softmax(Activation):
    """
    Softmax non-linearity.

    This activation is a vector activation (i.e., not elementwise).
    Unless for the output layer, be sure to check if
    the manual implementaion of model allows vector activations to be used
    in the pre-output layers in MLP.

    Attrs:
        A (np.ndarray): (batch size, num_features) Softmax(Z), values of activations
          each row sums to one.

    Methods:
        fordward
        backward
    """

    def forward(self, Z):
        """
        Args:
            Z (np.ndarray): (batch size, num_features)
        Returns:
            A (np.ndarray): (batch size, num_features) Softmax(Z), values of activations
            each row sums to one.
        """
        # self.softmax     = np.exp(Z)/ (np.exp(Z) @ Ones_C @ Ones_C.T)

        # log_sum trick 1
        # Z_tem = Z - np.amax(Z, axis=1).reshape((Z.shape[0],1))
        # self.A = np.exp(Z_tem)/np.sum(np.exp(Z_tem), axis=1, keepdims=True)

        # log_sum trick 2
        m = np.amax(Z, axis=1)
        Z_m = Z - m.reshape((Z.shape[0], 1))
        lse = np.log(np.sum(np.exp(Z_m), axis=1)) + m
        self.A = np.exp(Z - lse.reshape((Z.shape[0], 1)))  # softmax of Z

    def backward(self):
        # not yet implemented for vector activation.
        raise NotImplemented


class Tanh(Activation):

    """
    Tanh non-linearity (elementwise)

    Attrs:
        A: value of activation

    Methods:
        forward
        backward
    """

    def forward(self, Z):
        """
        Args:
            Z

        Returns:
            self.A
        """
        # self.A = np.sinh(Z) / np.cosh(Z)
        self.A = np.tanh(Z)

        return self.A

    def backward(self):
        dAdZ = 1 - np.power(self.A, 2)

        return dAdZ


class Tanh2(Activation):

    """
    Modified Tanh to work with BPTT.

    Now in the derivative case, we can pass in the stored hidden state and
    compute the derivative for that state instead of the "current" stored state
    self.A which could be different.

    See rnn_cell.py for details.
    """

    def __init__(self):
        super(Tanh2, self).__init__()

    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, state=None):
        # note that the earlier version of Tanh().backward does not allow argument.
        if state is not None:
            return 1 - (state**2)
        else:
            return 1 - (self.A**2)


class ReLU(Activation):

    """
    ReLU non-linearity (elementwise)

    Attrs:
        A: value of activation

    Methods
        forward
        backward
    """

    def forward(self, Z):
        """
        Args:
            Z

        Returns:
            self.A
        """
        self.A = np.maximum(Z, np.zeros(Z.shape, dtype="f"))
        # =np.clip(Z, a_min=0, a_max=None)

        return self.A

    def backward(self):
        dAdZ = np.where(self.A > 0, 1, 0)

        return dAdZ
