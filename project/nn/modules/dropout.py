import numpy as np


class Dropout(object):

    """
    Interface

    Note: This dropout laye is to be placed right after an activation.
    Ideally, it should not be applied to output layer.
    But one can apply it to the input layer, i.e., after an identity
    activation applied to input layer.

    Implemented as inverted dropout, i.e., no modification in the inference phase.

    Args:
        p (float): drop probability

    Attrs:
        p (float): drop probability
        mask (np.ndarray): (batch size, number of neurons)
                A matrix of Bernoulli random vairables.

    Methods:
        forward
        backward
    """

    def __init__(self, p=0.5):
        self.p = p  # p is drop probability
        self.mask = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):
        """
        The train parameter is to indicate whether we are in the
        training phase of the problem.

        Args:
            x (np.ndarray): (batch size, number of neurons)
                input to the dropout layer
            train: Boolean

        Returns:
            x_kept (np.ndarray): (batch size, number of neurons)
        """

        if train:
            # Generate mask and apply to x
            self.mask = np.random.binomial(n=1, p=1 - self.p, size=x.shape)
            x_kept = self.mask * x / (1 - self.p)
            return x_kept

        else:
            # Return x as is
            x
            return x

    def backward(self, delta):
        """
        Args:
            delta (np.ndarray): (batch size, number of neurons)
                delta is the dLdx_kept, x_kept is output from forward().

        Returns:
            delta_masked (np.ndarray): (batch size, number of neurons)
                the dLdx, x is input to forward().

        """

        # Multiply mask with delta and return
        delta_masked = self.mask * delta
        return delta_masked
