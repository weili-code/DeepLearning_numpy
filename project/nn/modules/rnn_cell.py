import numpy as np
from nn.modules.parameters import Parameter
from nn.modules.activation import *


class RNNCell(object):
    """
    A RNN Cell class. A single time stamp in a single layer.

    Here, we allow general batch size, and we define a single cell at one time stamp and one layer.

    Note that we use Tanh2() as the activation function.

    Args:
        input_size  (int): The dimensionality of the input vectors (at a single time).
        hidden_size (int): The dimensionality of the hidden state vectors (at a single time)..

    Attributes:
        W_ih (obj-Parameter <nn.modules.parameters>): Weight matrix (hidden_size, input_size) for input-to-hidden transformations.
        W_hh (obj-Parameter <nn.modules.parameters>): Weight matrix for hidden-to-hidden transformations.
        b_ih (obj-Parameter <nn.modules.parameters>): Bias vector (hidden_size, ) for input-to-hidden transformations.
        b_hh (obj-Parameter <nn.modules.parameters>): Bias vector for hidden-to-hidden transformations.
        activation (obj-nn.modules.activation): The activation function used within the RNN cell (Tanh2 in this case).
        parameters: a dict {"W_ih": W_ih, "W_hh": W_hh,  "b": b_ih, "b": b_hh}

        e.g.
        self.W_ih.data: returns the values of W_ih
        self.W_ih.grad: returns the values of gradient of W_ih
        self.b_ih.data: returns the values of b_ih
        self.b_hh.grad: returns the values of gradient of b_ih

    Methods:
        init_weights: Initialize the weight matrices and bias vectors.
        zero_grad: Reset the gradients of weight matrices and bias vectors to zero.
        forward: Forward pass for a single time step.
        backward: Backward pass for a single time step (Backpropagation Through Time for one step).
    """

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Activation function for
        self.activation = Tanh2()

        # hidden dimension and input dimension
        h = self.hidden_size
        d = self.input_size

        # Weights and biases (std normal initialization)
        self.W_ih = Parameter(np.random.randn(h, d))
        self.W_hh = Parameter(np.random.randn(h, h))
        self.b_ih = Parameter(np.random.randn(h))
        self.b_hh = Parameter(np.random.randn(h))

        self.parameters = {
            "W_ih": self.W_ih,
            "W_hh": self.W_hh,
            "b_ih": self.b_ih,
            "b_hh": self.b_hh,
        }

    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        """
        method to initialize weights values, i.e.,

        W_ih, W_hh, b_ih, b_hh np.arrays
        w_ih: shape (h, d)
        w_hh: shape (h, h)
        b_ih: shape (h, )
        b_hh: shape (h, )

        they are used to initialize W.data, and b.data.
        """
        self.W_ih.data = W_ih
        self.W_hh.data = W_hh
        self.b_ih.data = b_ih
        self.b_hh.data = b_hh

    def zero_grad(self):
        # already defined as a method in optimizer, so removed here
        d = self.input_size
        h = self.hidden_size
        self.W_ih.grad = np.zeros((h, d))
        self.W_hh.grad = np.zeros((h, h))
        self.b_ih.grad = np.zeros(h)
        self.b_hh.grad = np.zeros(h)

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """
        RNN Cell forward (single time step).

        Args (see writeup for explanation):
        x: (batch_size, input_size)
            input at the current time step

        h_prev_t: (batch_size, hidden_size)
            hidden state at the previous time step and current layer

        Returns:
        h_t: (batch_size, hidden_size)
            hidden state at the current time step and current layer

            h_t = tanh(Wih * xt + bih + Whh * h_{t−1} + bhh)
            recall:
                Wih (h, d); Whh (h, h)
                bih (h, )
        """

        h_t = self.activation(
            np.dot(x, self.W_ih.data.T)
            + self.b_ih.data
            + np.dot(h_prev_t, self.W_hh.data.T)
            + self.b_hh.data
        )

        # raise NotImplementedError
        return h_t

    def backward(self, delta, h_t, h_prev_l, h_prev_t):
        """
        RNN Cell backward (single time step). single time step BPTT.

        key equation: h_t = tanh(Wih * xt + bih + Whh * h_{t−1} + bhh)

        Args:
            delta: (batch_size, hidden_size)
                    Gradient w.r.t the current hidden layer

            h_t: (batch_size, hidden_size)
                    Hidden state of the current time step and the current layer

            h_prev_l: (batch_size, input_size)
                    Hidden state at the current time step and previous layer

            h_prev_t: (batch_size, hidden_size)
                    Hidden state at previous time step and current layer

        Returns:
            dx: (batch_size, input_size)
                Gradient w.r.t.  the current time step and previous layer

            dh_prev_t: (batch_size, hidden_size)
                Gradient w.r.t.  the previous time step and current layer


        Here, we allow general batch size, and we define a single cell at one time stamp and one layer.

        Note the backprop formulae here resembles that of the
        classical neural network neuron.

        Note also we use += for gradients for the parameters. So the same cell can be used across time
        to accumulate the gradients (BPTT).

        """

        batch_size = delta.shape[0]

        # 0) Step backward through the tanh activation function.
        # Note, because of BPTT, we had to externally save the tanh state, and
        # have modified the tanh activation function to accept an optionally input.
        # For example, the tanh states are saved via self.hiddens in the class RNNPhonemeClassifier
        # in the rnn_classifier.py.

        dz = self.activation.backward(state=h_t) * delta  # (batch_size, hidden_size)

        # 1) Compute the averaged gradients of the weights and biases
        # assuming the loss takes the form of cross entropy over batch size
        # note the accumulation += sign here. as the parameters do not vary with time

        self.W_ih.grad += (
            np.dot(dz.T, h_prev_l) / batch_size
        )  # (hidden_size, input_size)
        self.W_hh.grad += (
            np.dot(dz.T, h_prev_t) / batch_size
        )  # (hidden_size, hidden_size)
        self.b_ih.grad += np.mean(dz, axis=0)  # (hidden_size)
        self.b_hh.grad += np.mean(dz, axis=0)  # (hidden_size)

        # # 2) Compute dx, dh_prev_t
        # note that no averaging is needed, as they are the inputs to the cell.

        dx = np.dot(dz, self.W_ih.data)  # (batch_size, input_size)
        dh_prev_t = np.dot(dz, self.W_hh.data)  # (batch_size, hidden_size)

        # 3) Return dx, dh_prev_t
        return dx, dh_prev_t
