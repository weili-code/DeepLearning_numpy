import numpy as np


def label_to_num(targets, labels):
    """
    convert a numpy 1D array of string labels (targets) to an array of numerical values
    according to the labels array (list)

    e.g.
    labels = ["orange", "apple", "banana"]
    targets =  ["orange", "apple", "banana", "orange", "apple", "banana", ...]
    return [0, 1, 2, 0, 1, 2, ...]

    Args:
        targets (np 1d array): a numpy 1D array of string labels to be converted
        labels (list): a list of unique string labels (total K classes)

    Returns:
        np 1d array: array of targets in numerical values of 0, 1, ..., K-1
    """

    # Create a mapping dictionary from labels to numerical values
    label_to_num = {label: i for i, label in enumerate(labels)}

    # Convert the string labels to numerical values
    numerical_labels = np.array([label_to_num[y] for y in targets])

    return numerical_labels


def to_onehot(targets, num_classes):
    """
    Args:
        targets (np.ndarray): (batch size, 1), of class value 0, ..., K-1
        num_classes (int):

    Returns:
        out (np.ndarray): (batch size, num_classes)
    """
    # convert a vector of targets of class value 0, ..., K-1 to one-hot encoding
    # np.ndarray with shape=(targets.shape[0], K)
    out = np.zeros((targets.shape[0], num_classes))
    for row, col in enumerate(targets):
        out[row, col] = 1
    return out


class input_normalizer:
    """
    Standardize input data for np array data.
    nomalize using grand mean and std obtained from training data

    data has shape (batch size, num_features)

    Methods:
        fit_transform
        transform

    Returns:
        out (np.ndarray): (batch size, num_features)
    """

    def __init__(self):
        self.mu = None
        self.std = None
        self.transformed_x = None
        self.method = None

    def fit_transform(self, x, method):
        """
        transform the data by standardization, save mean and std as attributes.

        Args:
            x (np.ndarray): (batch size, num_features)
            method (str): method for normalization
                "grand": normalization by grand mean and standard deviation
                "column": normalization by mean and standard deviation for each column

        Attrs:
            data (np.ndarray): (batch size, num_features). original x
            mu (scalar): grand mean of all entries of x
            std (scalar): grand std of all entries of x
            transform_x (np.ndarray): (batch size, num_features).

        Returns:
            transform_x (np.ndarray): (batch size, num_features)
        """

        self.data = x
        self.method = method
        if method == "grand":
            # normalization by grand mean and standard deviation
            # self.mu, self.std = np.mean(self.data, axis=0, keepdims=True), np.std(self.data, axis=0, keepdims=True)
            self.mu, self.std = np.mean(self.data), np.std(self.data)
        if method == "column":
            # normalization by mean and standard deviation for each column
            self.mu, self.std = np.mean(self.data, axis=0), np.std(self.data, axis=0)
            # mu and std (num_features, )
        self.transformed_x = (self.data - self.mu) / self.std
        return self.transformed_x

    def transform(self, newx):
        """
        transform the data by standardization, using the mean and std attributes.

        Args:
            newx (np.ndarray): (batch size, num_features)

        Returns:
             (np.ndarray): (batch size, num_features)
        """
        return (newx - self.mu) / self.std


class input_normalizer2:
    """
    data has shape (batch size, num_frames/num_channels, num_features)
    """

    def __init__(self):
        self.mu = None
        self.std = None
        self.transformed_x = None
        self.method = None

    def fit_transform(self, x, method):
        self.data = x
        self.method = method
        if method == "grand":
            # Normalization by grand mean and standard deviation
            # The mean and std are calculated over all dimensions
            self.mu, self.std = np.mean(self.data), np.std(self.data)
        elif method == "column":
            # Normalization by mean and standard deviation for each feature.
            # Axis 0 and 1 are the num_examples and num_frames dimensions,
            # Axis 2 is the num_features dimension
            self.mu, self.std = np.mean(self.data, axis=(0, 1), keepdims=True), np.std(
                self.data, axis=(0, 1), keepdims=True
            )

        # Broadcasting is used here to match the shapes for subtraction and division
        self.transformed_x = (self.data - self.mu) / self.std
        return self.transformed_x

    def transform(self, newx):
        return (newx - self.mu) / self.std
