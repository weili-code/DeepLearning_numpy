import numpy as np


class SgdDecay:

    """
    Interface
    SGD with L2 regularization and momentum.
    Note momentum method will take into account the effect from L2 regularization.

    Args:
        model_paras (list): list of model parameters
                        (Parameter <nn.modules.parameters> objects)
        lr (float):  Defaults to 0.1.
        momentum (float): Defaults to 0.
        l2_weight (float): Defaults to 0. l2 penalizing coefficient

    Attrs:
        l (list): list of model parameters
                (Parameter <nn.modules.parameters> objects)

        L (int): length of l, i.e., number of layers with parameters
        lr (float): learning rate
        mu (float): momentum parameter
        l2_weight (float): l2 penalizing coefficient
        v_W (list): a list of momentum for W (length = L)
        v_b (list): a list of momentum for b (length = L)


    Methods:

        step

    """

    def __init__(self, model_paras, lr=0.1, momentum=0.0, l2_weight=0.0):
        self.l = model_paras  # a list of model parameters
        self.L = len(model_paras)  # number of layers of parameters
        self.lr = lr  # learning rate
        self.mu = momentum  # momentum parameter
        self.l2_weight = l2_weight

        self.v_W = [
            np.zeros(self.l[i]["W"].data.shape, dtype="f") for i in range(self.L)
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
        step(): do one update on parameters
        """

        for i in range(self.L):
            dw = self.l[i]["W"].grad
            db = self.l[i]["b"].grad

            if self.l2_weight != 0:
                dw += self.l2_weight * self.l[i]["W"].data
                db += self.l2_weight * self.l[i]["b"].data

            if self.mu == 0:
                self.l[i]["W"].data = self.l[i]["W"].data - self.lr * dw
                self.l[i]["b"].data = self.l[i]["b"].data - self.lr * db
            else:
                # with momentum
                self.v_W[i] = self.mu * self.v_W[i] + dw
                self.v_b[i] = self.mu * self.v_b[i] + db
                self.l[i]["W"].data = self.l[i]["W"].data - self.lr * self.v_W[i]
                self.l[i]["b"].data = self.l[i]["b"].data - self.lr * self.v_b[i]

        return None
