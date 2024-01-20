import numpy as np


class Criterion(object):
    """
    Interface for loss functions.

    Attrs:
        A (np.ndarray): (batch size, number of output)
            logits if used for linear layer
        Y (np.ndarray): (batch size, number of output)
           actual output
        loss (np.ndarray): (scalar)

    Methods:
        forward
        backward
    """

    def __init__(self):
        self.A = None  # logits
        self.Y = None  # true output
        self.loss = None  # loss

    def __call__(self, A, Y):
        return self.forward(A, Y)

    def forward(self, A, Y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class MSELoss(Criterion):

    """
    Class for implementing MLELoss, allowing multivariate output.

    Attrs:
        A (np.ndarray): (batch size, number of output)
        Y (np.ndarray): (batch size, number of output)
           actual output (allowing multivariate output)
        loss (np.ndarray): (scalar) loss averaged over batchsize.

    Methods:
        forward
        backward

    """

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, A, Y):
        """
        Args:
            A (np.ndarray): (batch size, number of output)
            Y (np.ndarray): (batch size, number of output)

        Returns:
            loss (np.float): (scalar)
        """

        self.A = A
        self.Y = Y
        N = A.shape[0]
        C = A.shape[1]
        se = (self.A - self.Y) * (self.A - self.Y)
        sse = np.ones((N, 1), dtype="f").T @ se @ np.ones((C, 1), dtype="f")
        self.loss = sse / (N * C)  # mse

        return self.loss

    def backward(self):
        """
        Returns:
            dLdA (np.ndarray): (batch size, number of output)
                gradient of loss wrt A
        """
        dLdA = self.A - self.Y

        return dLdA


class CrossEntropyLoss(Criterion):

    """
    Class for implementing Cross Entropy Loss
    Subclass of Criterion

    Attrs:
        A (np.ndarray): (batch size, num_classes) logits
        Y (np.ndarray): (batch size, num_classes)
                        labels in one-hot encoding
        softmax (np.ndarray): (batch size, num_classes)
            each row adds up to 1 (total probability)
        loss (np.float): loss (averaged over) batchsize


    """

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.softmax = None

    def forward(self, A, Y):
        """
        Args:
            A: np.ndarray (batch size, num_classes)
            Y: np.ndarray (batch size, num_classes)
        Returns:
            loss: (np.float)
        """

        self.A = A
        self.Y = Y
        N = A.shape[0]
        C = A.shape[1]
        Ones_C = np.ones((C, 1), dtype="f")
        Ones_N = np.ones((N, 1), dtype="f")

        # self.softmax     = np.exp(self.A)/ (np.exp(self.A) @ Ones_C @ Ones_C.T)

        # log_sum trick 1
        # A_tem = self.A - np.amax(self.A, axis=1).reshape((self.A.shape[0],1))
        # self.softmax = np.exp(A_tem)/np.sum(np.exp(A_tem), axis=1, keepdims=True)

        # log_sum trick 2
        m = np.amax(self.A, axis=1)
        A_m = self.A - m.reshape((self.A.shape[0], 1))
        lse = np.log(np.sum(np.exp(A_m), axis=1)) + m
        self.softmax = np.exp(self.A - lse.reshape((self.A.shape[0], 1)))

        crossentropy = -self.Y * np.log(self.softmax)
        sum_crossentropy = Ones_N.T @ crossentropy @ Ones_C
        self.loss = np.asscalar(sum_crossentropy / N)

        return self.loss

    def backward(self):
        """
        Returns:
            dLdA (np.ndarray): (batch size, num_classes)
        """
        dLdA = self.softmax - self.Y

        return dLdA


class CrossEntropyLoss2(Criterion):

    """
    Class for implementing Cross Entropy Loss
    Subclass of Criterion

    Note: this class is difference from CrossEntropyLoss(Criterion)
    in that the returned loss from forward() is an array

    Attrs:
        A (np.ndarray): (batch size, num_classes) logits
        Y (np.ndarray): (batch size, num_classes)
                        labels in one-hot encoding
        softmax (np.ndarray): (batch size, num_classes)
            each row adds up to 1 (total probability)
        loss (np.ndarra):  (batch size, ) loss for each obs in the batch


    """

    def __init__(self):
        super(CrossEntropyLoss2, self).__init__()
        self.softmax = None

    def forward(self, A, Y):
        """
        Argument:
            A (np.array): (batch size, num_classses 10): logits
            Y (np.array): (batch size, num_classes 10): labels
        Return:
            out (np.array): (batch size, ): loss for cross entropy (not averaged)
        """

        self.logits = A
        self.labels = Y
        self.batch_size = self.labels.shape[0]
        exps = np.exp(self.logits)
        self.softmax = exps / exps.sum(axis=1, keepdims=True)
        self.loss = np.sum(np.multiply(self.labels, -np.log(self.softmax)), axis=1)

        return self.loss

    def backward(self):
        """
        Returns:
            dLdA (np.ndarray): (batch size, num_classes)
        """
        dLdA = self.softmax - self.labels

        return dLdA
