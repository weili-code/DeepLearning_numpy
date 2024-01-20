import numpy as np
from nn.modules.parameters import Parameter


class Linear(object):
    """
    Class for a linear layer.

    Z = A @ self.W.T + self.Ones @ self.b.T

    With the following dimensions
    A:  (batch size, in)
    W:  (out, in)
    b:  (out, 1)
    Z:  (batch size, out)

    Note in the code, we accumulate gradients for W and b.

    Args:
        in_features  (int): size of input
        out_features (int): size of output
        weight_init_fn (func): initializer for W
        bias_init_fn (func): initializer for b
        debug (bool, optional): _description_. Defaults to False.

    Attrs:
        W (obj-Parameter <nn.modules.parameters>): (out_features, in_features)
        b (obj-Parameter <nn.modules.parameters>): (out_features, 1)
            note: Pytorch nn.module, bias is of shape (out_features, )
        parameters: a dict {"W": W, "b": b}

        self.W.data: returns the values of W
        self.W.grad: returns the values of dW
        self.b.data: returns the values of b
        self.b.grad: returns the values of db

    Methods:
        forward
        backward
    """

    def __init__(
        self,
        in_features,
        out_features,
        weight_init_fn=None,
        bias_init_fn=None,
        debug=False,
    ):
        self.W = Parameter(np.zeros((out_features, in_features)))

        self.b = Parameter(np.zeros((out_features, 1)))

        self.debug = debug

        if weight_init_fn is not None:
            self.W.data = weight_init_fn(self.W.data)
        if bias_init_fn is not None:
            self.b.data = bias_init_fn(self.b.data)

        self.parameters = {"W": self.W, "b": self.b}

    def __call__(self, A):
        return self.forward(A)

    def forward(self, A):
        """
        Args:
            A: np.array (batch size, in_features)

        Returns:
            Z: np.array (batch size, out_features)
        """

        self.A = A
        self.N = A.shape[0]
        self.Ones = np.ones((self.N, 1))
        Z = A @ self.W.data.T + self.Ones @ self.b.data.T

        return Z

    def backward(self, dLdZ):
        """
        backward(dLdZ) populates dLdW, dLdb and returns dLdA
        i.e., dLdZ --> (dLdW, dLdb)-->dLdA
        (dLdW, dLdb) are populated as attributes

        Note: the accumulated gradients!

        Args:
            dLdZ (np.array): (batch size, out_features)

        Attrs:
            dLdW (np.array): (out_features, in_features)
            dLdb (np.array): (out_features, 1)

        Returns:
            dLdA (np.array): (batch size, in_features)
        """

        dZdA = self.W.data
        dZdW = self.A
        dZdi = None
        dZdb = self.Ones
        dLdA = dLdZ @ dZdA
        dLdW = dLdZ.T @ dZdW
        dLdi = None
        dLdb = dLdZ.T @ dZdb
        self.W.grad += dLdW / self.N  # note the accumulated gradients!
        self.b.grad += dLdb / self.N

        if self.debug:
            self.dZdA = dZdA
            self.dZdW = dZdW
            self.dZdi = dZdi
            self.dZdb = dZdb
            self.dLdA = dLdA
            self.dLdi = dLdi

        return dLdA
