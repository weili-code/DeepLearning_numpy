import numpy as np
from nn.modules.parameters import Parameter


class BatchNorm1d(object):

    """
    Interface
    Note: we accumulate gradients for W and b (batchnorm parameters).
    This BatchNorm1d layer is applied before the activation function.
    BatchNorm1d is a vector actication layer.

    The BN layer acts like this:
    Z-->NZ-->BZ

    Args:
        num_features (int) :number of neurons (Batchnorm is applied to each layer, neuron-wise)
        alpha (float): decay coefficient for the running average of mean and variances
        used in the inference stage
        weight_init_fn (func): initializer for W
        bias_init_fn (func): initializer for b

    Attrs:
        alpha (float): decay coefficient
        eps (float): stabalizing error
        Z (np.ndarray): (batch size, num_features)
                        data input to the Batchnorm layer
        NZ (np.ndarray): (batch size, num_features)
                        normalized Z
        BZ (np.ndarray): (batch size, num_features)
                        returned transformed Z
        BW (Parameter <nn.modules.parameters> obj): (1, num_features)
                        weight parameter in the batchnorm layer
        Bb (Parameter <nn.modules.parameters> obj): np.array (1, num_features)
                        bias parameter in the batchnorm layer
        M (np.ndarray): (1, num_features)
                    per feature(neuron) mean vector
        V (np.ndarray): (1, num_features)
                        per feature(neuron) variance vector
        running_M (np.ndarray): (1, num_features)
        running_V (np.ndarray): (1, num_features)

    Methods:
        forward
        backward
    """

    def __init__(self, num_features, weight_init_fn=None, bias_init_fn=None, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8

        self.Z = None
        self.NZ = None
        self.BZ = None

        self.BW = Parameter(np.ones((1, num_features)))
        self.Bb = Parameter(np.zeros((1, num_features)))

        if weight_init_fn is not None:
            self.BW.data = weight_init_fn(self.BW.data)
        if bias_init_fn is not None:
            self.Bb.data = bias_init_fn(self.Bb.data)

        self.M = np.zeros((1, num_features))
        self.V = np.ones((1, num_features))

        # inference parameters
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

        self.parameters = {"W": self.BW, "b": self.Bb}

    def forward(self, Z, train=True):
        """
        The train parameter is to indicate whether we are in the
        training phase of the problem or are we in the inference phase.

        Args:
            Z (np.ndarray): (batch size, num_features)

        Returns:
            BZ (np.ndarray): (batch size, num_features)
        """

        if not train:
            self.NZ = (Z - self.running_M) / np.sqrt(self.running_V + self.eps)
            self.BZ = self.BW.data * self.NZ + self.Bb.data
            return self.BZ

        self.Z = Z
        self.N = self.Z.shape[0]

        self.M = np.mean(self.Z, axis=0)
        self.V = np.var(self.Z, axis=0)
        self.NZ = (self.Z - self.M) / np.sqrt(self.V + self.eps)
        self.BZ = self.BW.data * self.NZ + self.Bb.data

        self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
        self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V

        return self.BZ

    def backward(self, dLdBZ):
        """
        Note that we accumulate gradients.

        Args:
            dLdBZ (np.ndarray): (batch size, num_features)

        Returns:
            dLdZ (np.ndaarray): (batch size, num_features)
        """

        self.BW.grad += np.sum(
            dLdBZ * self.NZ, axis=0, keepdims=True
        )  # note here accumulated gradients!
        self.Bb.grad += np.sum(dLdBZ, axis=0, keepdims=True)

        dLdNZ = dLdBZ * self.BW.data

        Z_sub_M = self.Z - self.M
        sqrt_V = np.sqrt(self.V + self.eps)
        dLdV = -np.sum(dLdNZ * (Z_sub_M / (2 * (sqrt_V**3))), axis=0)

        N = dLdBZ.shape[0]
        dLdM = -np.sum(dLdNZ / sqrt_V, axis=0) - (2 / N) * dLdV * np.sum(
            Z_sub_M, axis=0
        )

        dLdZ = (dLdNZ / sqrt_V) + (dLdV * (2 / N) * Z_sub_M) + (dLdM / N)

        return dLdZ
