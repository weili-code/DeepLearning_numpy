import numpy as np

# this script is a modified adam.py for implementing RNN
# Specifically, it modifies 'W' and 'b' to 'W_ih', 'W_hh', 'b_ih', 'b_hh'.


class Adam_rnn:

    """
    Interface

    Args:
        model_paras (list): list of model parameters
                        (Parameter <nn.modules.parameters> objects)
        lr (float): learning rate
        beta1 (float): decaying rate for running average of gradient
        beta2 (float): decaying rate for running average of squared gradient
        eps (float)

    Attrs:
        l (list): list of model parameters
                (Parameter <nn.modules.parameters> objects)
        L (int): length of l, i.e., number of layers with parameters
        beta1 (float): decaying rate for running average of gradient
        beta2 (float): decaying rate for running average of squared gradient
        lr (float): learning rate
        eps (float): stablizing corrector
        t (int): time-stamp tracking update iterate, accrue for each step()
        m_W (float): running average of gradient
        m_b
        v_W (float): running average of squared gradient
        v_b

    ---------
    METHODS

    step

    """

    def __init__(self, model_paras, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.l = model_paras  # a list of model parameters
        self.L = len(model_paras)  # number of layers of parameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = lr
        self.t = 0

        self.m_W_ih = [
            np.zeros(self.l[i]["W_ih"].data.shape, dtype="f") for i in range(self.L)
        ]
        self.v_W_ih = [
            np.zeros(self.l[i]["W_ih"].data.shape, dtype="f") for i in range(self.L)
        ]

        self.m_b_ih = [
            np.zeros(self.l[i]["b_ih"].data.shape, dtype="f") for i in range(self.L)
        ]
        self.v_b_ih = [
            np.zeros(self.l[i]["b_ih"].data.shape, dtype="f") for i in range(self.L)
        ]

        self.m_W_hh = [
            np.zeros(self.l[i]["W_hh"].data.shape, dtype="f") for i in range(self.L)
        ]
        self.v_W_hh = [
            np.zeros(self.l[i]["W_hh"].data.shape, dtype="f") for i in range(self.L)
        ]

        self.m_b_hh = [
            np.zeros(self.l[i]["b_hh"].data.shape, dtype="f") for i in range(self.L)
        ]
        self.v_b_hh = [
            np.zeros(self.l[i]["b_hh"].data.shape, dtype="f") for i in range(self.L)
        ]

    def zero_grad(self):
        for i in range(self.L):
            if self.l[i]["W_ih"].grad is not None:
                self.l[i]["W_ih"].grad = np.zeros(self.l[i]["W_ih"].grad.shape)
            if self.l[i]["b_ih"].grad is not None:
                self.l[i]["b_ih"].grad = np.zeros(self.l[i]["b_ih"].grad.shape)
            if self.l[i]["W_hh"].grad is not None:
                self.l[i]["W_hh"].grad = np.zeros(self.l[i]["W_hh"].grad.shape)
            if self.l[i]["b_hh"].grad is not None:
                self.l[i]["b_hh"].grad = np.zeros(self.l[i]["b_hh"].grad.shape)

    def step(self):
        """
        step(): do one update on weights and biases in each layer of the model
        """

        self.t += 1
        for i in range(self.L):
            self.m_W_ih[i] = (
                self.beta1 * self.m_W_ih[i] + (1 - self.beta1) * self.l[i]["W_ih"].grad
            )
            self.v_W_ih[i] = (
                self.beta2 * self.v_W_ih[i]
                + (1 - self.beta2) * (self.l[i]["W_ih"].grad) ** 2
            )
            self.m_b_ih[i] = (
                self.beta1 * self.m_b_ih[i] + (1 - self.beta1) * self.l[i]["b_ih"].grad
            )
            self.v_b_ih[i] = (
                self.beta2 * self.v_b_ih[i]
                + (1 - self.beta2) * (self.l[i]["b_ih"].grad) ** 2
            )

            self.m_W_hh[i] = (
                self.beta1 * self.m_W_hh[i] + (1 - self.beta1) * self.l[i]["W_hh"].grad
            )
            self.v_W_hh[i] = (
                self.beta2 * self.v_W_hh[i]
                + (1 - self.beta2) * (self.l[i]["W_hh"].grad) ** 2
            )
            self.m_b_hh[i] = (
                self.beta1 * self.m_b_hh[i] + (1 - self.beta1) * self.l[i]["b_hh"].grad
            )
            self.v_b_hh[i] = (
                self.beta2 * self.v_b_hh[i]
                + (1 - self.beta2) * (self.l[i]["b_hh"].grad) ** 2
            )

            m_W_ih_tilde = self.m_W_ih[i] / (1 - self.beta1**self.t)
            v_W_ih_tilde = self.v_W_ih[i] / (1 - self.beta2**self.t)
            m_b_ih_tilde = self.m_b_ih[i] / (1 - self.beta1**self.t)
            v_b_ih_tilde = self.v_b_ih[i] / (1 - self.beta2**self.t)

            m_W_hh_tilde = self.m_W_hh[i] / (1 - self.beta1**self.t)
            v_W_hh_tilde = self.v_W_hh[i] / (1 - self.beta2**self.t)
            m_b_hh_tilde = self.m_b_hh[i] / (1 - self.beta1**self.t)
            v_b_hh_tilde = self.v_b_hh[i] / (1 - self.beta2**self.t)

            # Calculate updates for weight
            self.l[i]["W_ih"].data = self.l[i][
                "W_ih"
            ].data - self.lr * m_W_ih_tilde / np.sqrt(v_W_ih_tilde + self.eps)

            # calculate updates for bias
            self.l[i]["b_ih"].data = self.l[i][
                "b_ih"
            ].data - self.lr * m_b_ih_tilde / np.sqrt(v_b_ih_tilde + self.eps)

            self.l[i]["W_hh"].data = self.l[i][
                "W_hh"
            ].data - self.lr * m_W_hh_tilde / np.sqrt(v_W_hh_tilde + self.eps)

            # calculate updates for bias
            self.l[i]["b_hh"].data = self.l[i][
                "b_hh"
            ].data - self.lr * m_b_hh_tilde / np.sqrt(v_b_hh_tilde + self.eps)

        return None
