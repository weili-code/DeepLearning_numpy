import numpy as np


class Adam:

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

    Methods:

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

        self.m_W = [
            np.zeros(self.l[i]["W"].data.shape, dtype="f") for i in range(self.L)
        ]
        self.v_W = [
            np.zeros(self.l[i]["W"].data.shape, dtype="f") for i in range(self.L)
        ]

        self.m_b = [
            np.zeros(self.l[i]["b"].data.shape, dtype="f") for i in range(self.L)
        ]
        self.v_b = [
            np.zeros(self.l[i]["b"].data.shape, dtype="f") for i in range(self.L)
        ]

    def zero_grad(self):
        for i in range(self.L):
            if self.l[i]["W"].grad is not None:
                self.l[i]["W"].grad = np.zeros(self.l[i]["W"].grad.shape)
            if self.l[i]["b"].grad is not None:
                self.l[i]["b"].grad = np.zeros(self.l[i]["b"].grad.shape)

    def step(self):
        """
        step(): do one update on weights and biases in each layer of the model
        """

        self.t += 1
        for i in range(self.L):
            self.m_W[i] = (
                self.beta1 * self.m_W[i] + (1 - self.beta1) * self.l[i]["W"].grad
            )
            self.v_W[i] = (
                self.beta2 * self.v_W[i] + (1 - self.beta2) * (self.l[i]["W"].grad) ** 2
            )
            self.m_b[i] = (
                self.beta1 * self.m_b[i] + (1 - self.beta1) * self.l[i]["b"].grad
            )
            self.v_b[i] = (
                self.beta2 * self.v_b[i] + (1 - self.beta2) * (self.l[i]["b"].grad) ** 2
            )

            m_W_tilde = self.m_W[i] / (1 - self.beta1**self.t)
            v_W_tilde = self.v_W[i] / (1 - self.beta2**self.t)
            m_b_tilde = self.m_b[i] / (1 - self.beta1**self.t)
            v_b_tilde = self.v_b[i] / (1 - self.beta2**self.t)

            # calculate updates for weight
            self.l[i]["W"].data = self.l[i]["W"].data - self.lr * m_W_tilde / np.sqrt(
                v_W_tilde + self.eps
            )

            # calculate updates for bias
            self.l[i]["b"].data = self.l[i]["b"].data - self.lr * m_b_tilde / np.sqrt(
                v_b_tilde + self.eps
            )

        return None


class AdamW:
    """
    Interface
    Recall the Adam, update
        W_new = W_old -  lr * adjusted m_W
    Similar to Adam, except that in the last update, AdamW do
    W_new = W_old -lr * weight_decay * W_old -  lr * adjusted m_W

    Args:
        model_paras (list): list of model parameters
                        (Parameter <nn.modules.parameters> objects)
        lr (float, optional): learning rate Defaults to 0.1.
        beta1 (float): decaying rate for running average of gradient
        beta2 (float): decaying rate for running average of squared gradient
        eps (float)
        weight_decay (float): default to 0.01.

    Attrs:
        l (list):  list of model parameters
                        (Parameter <nn.modules.parameters> objects)
        beta1 (float): decaying rate for running average of gradient
        beta2 (float): decaying rate for running average of squared gradient
        lr (float): learning rate
        eps (float): stablizing corrector
        weight_decay (float):
        t (int): time-stamp tracking update iterate, accrue for each step()
        m_W (float): running average of gradient
        m_b
        v_W (float): running average of squared gradient
            v_b

    Methods:
        step
        zero_grad
    """

    def __init__(
        self, model_paras, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01
    ):
        self.l = model_paras  # a list of model parameters
        self.L = len(model_paras)  # number of layers of parameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = lr
        self.t = 0
        self.weight_decay = weight_decay

        self.m_W = [
            np.zeros(self.l[i]["W"].data.shape, dtype="f") for i in range(self.L)
        ]
        self.v_W = [
            np.zeros(self.l[i]["W"].data.shape, dtype="f") for i in range(self.L)
        ]

        self.m_b = [
            np.zeros(self.l[i]["b"].data.shape, dtype="f") for i in range(self.L)
        ]
        self.v_b = [
            np.zeros(self.l[i]["b"].data.shape, dtype="f") for i in range(self.L)
        ]

    def zero_grad(self):
        for i in range(self.L):
            if self.l[i]["W"].grad is not None:
                self.l[i]["W"].grad = np.zeros(self.l[i]["W"].grad.shape)
            if self.l[i]["b"].grad is not None:
                self.l[i]["b"].grad = np.zeros(self.l[i]["b"].grad.shape)

    def step(self):
        self.t += 1
        for i in range(self.L):
            self.m_W[i] = (
                self.beta1 * self.m_W[i] + (1 - self.beta1) * self.l[i]["W"].grad
            )
            self.v_W[i] = (
                self.beta2 * self.v_W[i] + (1 - self.beta2) * (self.l[i]["W"].grad) ** 2
            )
            self.m_b[i] = (
                self.beta1 * self.m_b[i] + (1 - self.beta1) * self.l[i]["b"].grad
            )
            self.v_b[i] = (
                self.beta2 * self.v_b[i] + (1 - self.beta2) * (self.l[i]["b"].grad) ** 2
            )

            m_W_tilde = self.m_W[i] / (1 - self.beta1**self.t)
            v_W_tilde = self.v_W[i] / (1 - self.beta2**self.t)
            m_b_tilde = self.m_b[i] / (1 - self.beta1**self.t)
            v_b_tilde = self.v_b[i] / (1 - self.beta2**self.t)

            # Calculate updates for weight
            W_update = m_W_tilde / np.sqrt(v_W_tilde + self.eps)

            # calculate updates for bias
            b_update = m_b_tilde / np.sqrt(v_b_tilde + self.eps)

            # Perform weight and bias updates with weight decay
            self.l[i]["W"].data = (
                self.l[i]["W"].data
                - self.lr * W_update
                - self.lr * self.weight_decay * self.l[i]["W"].data
            )
            self.l[i]["b"].data = (
                self.l[i]["b"].data
                - self.lr * b_update
                - self.lr * self.weight_decay * self.l[i]["b"].data
            )

        return None
