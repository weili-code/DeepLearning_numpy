import numpy as np
import os
import sys
import pdb

sys.path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nn.modules.loss import *
from nn.modules.activation import *
from nn.modules.batchnorm import *
from nn.modules.linear import *
from nn.modules.dropout import *
from nn.modules.conv import *
from nn.modules.initializer import *
from utils import data_processor
from collections import OrderedDict


def print_dict(dic):
    inner_lines = "\n".join("%s:%s" % (k, v) for k, v in dic.items())
    return inner_lines


def print_keys(dic, indent=0):
    # print the keys of nested dictionary
    for key, value in dic.items():
        print("\t" * indent + str(key))
        if isinstance(value, dict):
            print_keys(value, indent + 2)


class CNN1D(object):
    """
    A simple convolutional neural network (1D)

    x->Conv1D (out_channels, kernerl, stride)-> activations -> pooling (if applicable)
     -> ...... repeat.....
     -> Flatten1D -> Linear -> output (logits)

    We uses CrossEntropyLoss, thus the output is logits.

    Example: RAVDESS speech data

    x (batch_size, in_channels=1, in_width=180)
    Conv1D (out_channels=8, kernel=10, stride=1)-> Tanh
    pool (kernel=2)
    Conv1D (out_channels=8, kernel=10, stride=2)-> ReLu
    pool (kernel=2)
    Conv1D (out_channels=4, kernel=4, stride=2) -> Sigmoid
    pool (kernel=1)
    Flatten1D
    Linear (out_features=8)


    out_channels = [8, 8, 4]
    kernel_sizes = [10, 10, 4]
    strides = [1, 2, 2]
    pool_kernel_sizes =[2, 2, 1]
    activations = [Tanh(), ReLU(), Sigmoid()]
    num_linear_neurons = 8

    Args:
        input_width           (int)    : The width of the input to the first convolutional layer
        num_input_channels    (int)    : Number of channels for the input layer
        num_channels          (list/int)  : List containing number of (output) channels for each conv layer
        kernel_sizes          (list/int)  : List containing kernel width for each conv layer
        strides               (list/int)  : List containing stride size for each conv layer
        num_linear_neurons    (int)    : Number of neurons (outputs) in the (last) linear layer
        activations           (list/obj-Activation <nn.modules.activation>)  : List of objects corresponding to the activation fn for each conv layer
        conv_weight_init_fn   (fn)     : Function to init each conv layers weights
        bias_init_fn          (fn)     : Function to initialize each conv layers AND the linear layers bias to 0
        linear_weight_init_fn (fn)     : Function to initialize the linear layers weights
        pool_kernel_sizes     (list/int)  : List containing kernel width for each pooling layer
        pool_mode             (str)    : "max" or "average" (default "max")

        note: if pooling layer is used, we require it to be applied after the activation of each conv layer,
        i.e., len(pool_kernel_sizes)=len(num_channels)
        and we use default stride for pooling layer, i.e., stride for pooling = kernel size,
        and pooling does not change channels numbers.
        Though not efficient, setting pool kernel size=1, is equivalent to not pooling at all.

    Attrs:
        train_mode: default=True
        num_conv_layers: number of conv layers
        num_pool_layers: number of pooling layers
        activations: a list of activations
        convolutional_layers (list/obj-Conv1D <nn.modules.conv>): a list of Conv1D objects
        pool_layers (list/obj-Pool1D <nn.modules.conv>): a list of Pool1D objects
        flatten (fn):  Flatten1D()
        linear_layer: (obj-Linear <nn.modules.linear>)
        nlayers: number of total layers ( number of conv layers + number of pool layers
                                        + 2, i.e., adding a flatten layer and linear output layer )
        layers_dict (OrderedDict of List):
            a ordered dict for layers: layers_dict
            
        paras_dict (Nested OrderedDict):
            dict for parameters of the model: paras_dict

    Methods:
        train
        eval
        forward
        backward
        print_structure
        get_parameters


    You can be sure that len(activations) == len(num_channels) == len(kernel_sizes) == len(strides)
    """

    def __init__(
        self,
        input_width,
        num_input_channels,
        num_channels,
        kernel_sizes,
        strides,
        num_linear_neurons,
        activations,
        conv_weight_init_fn,
        bias_init_fn,
        linear_weight_init_fn,
        pool_kernel_sizes=None,
        pool_mode="max",
    ):
        self.train_mode = True
        self.num_conv_layers = len(num_channels)
        if pool_kernel_sizes is not None:
            self.num_pool_layers = len(pool_kernel_sizes)
        self.activations = activations
        self.flatten = None
        self.linear_layer = None
        self.nlayers = self.num_conv_layers + self.num_pool_layers + 2

        outChannel = num_input_channels
        outSize = 0
        inputSize = input_width

        self.layers_dict = OrderedDict()
        self.paras_dict = OrderedDict()

        self.convolutional_layers = []
        self.pool_layers = []
        for i in range(self.num_conv_layers):
            self.convolutional_layers.append(
                Conv1D(
                    in_channel=outChannel,
                    out_channel=num_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    weight_init_fn=conv_weight_init_fn,
                    bias_init_fn=bias_init_fn,
                )
            )
            outChannel = num_channels[i]
            outSize = (inputSize - kernel_sizes[i]) // strides[i] + 1
            inputSize = outSize
            # if there is pooling layers
            if pool_kernel_sizes is not None:
                self.pool_layers.append(
                    Pool1D(pool_size=pool_kernel_sizes[i], mode=pool_mode)
                )
                outSize = (inputSize - pool_kernel_sizes[i]) // pool_kernel_sizes[i] + 1
                inputSize = outSize

        self.flatten = Flatten1D()

        self.linear_layer = Linear(
            in_features=outChannel * inputSize,
            out_features=num_linear_neurons,
            weight_init_fn=linear_weight_init_fn,
            bias_init_fn=bias_init_fn,
        )

        # create a ordered dict for layers: layers_dict

        # For sublayer that is Linear, BatchNorm1d or Conv objects, there is a
        # method parameters() to call parameters which is a dict with keys
        # 'W' and 'b'.
        idx_conv = 0
        idx_pool = 0
        for i in range(self.nlayers - 2):
            self.layers_dict["layer" + str(i)] = []
            if i % 2 == 0:
                self.layers_dict["layer" + str(i)].append(
                    self.convolutional_layers[idx_conv]
                )
                self.layers_dict["layer" + str(i)].append(self.activations[idx_conv])
                idx_conv += 1
            if (len(self.pool_layers) != 0) & (i % 2 != 0):
                self.layers_dict["layer" + str(i)].append(self.pool_layers[idx_pool])
                idx_pool += 1

        self.layers_dict["layer" + str(self.nlayers - 2)] = []
        self.layers_dict["layer" + str(self.nlayers - 2)].append(self.flatten)
        self.layers_dict["layer" + str(self.nlayers - 1)] = []
        self.layers_dict["layer" + str(self.nlayers - 1)].append(self.linear_layer)

        ## construct dict for parameters of the model: "paras_dict"

        for i in range(self.nlayers):
            layer = self.layers_dict["layer" + str(i)]
            self.paras_dict["layer" + str(i)] = {}
            sublayer_idx = 0
            for sublayer in layer:
                name = "(" + str(sublayer_idx) + ")"
                if isinstance(sublayer, Conv1D):
                    self.paras_dict["layer" + str(i)][
                        name + "conv1d"
                    ] = sublayer.parameters
                if isinstance(sublayer, Linear):
                    self.paras_dict["layer" + str(i)][
                        name + "linear"
                    ] = sublayer.parameters
                sublayer_idx += 1

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        inner_lines = "\n".join("%s:%s" % (k, v) for k, v in self.layers_dict.items())
        return inner_lines

    def train(self):
        # Do not modify this method
        self.train_mode = True

    def eval(self):
        # Do not modify this method
        self.train_mode = False

    def print_structure(self):
        """
        Print the structure of layers_dict and paras_dict
        """
        print("-----------------------")
        print("The model architecture:")
        for layer_key, layer_value in self.layers_dict.items():
            # Print the current layer name
            print(f"{layer_key}:")
            # Iterate through the sublayers and print them with indentation
            for i, element in enumerate(layer_value):
                sublayer_name = f"sublayer{i}"
                print(f"\t{sublayer_name}: {element}")
        print()
        print("---------------------------------")
        print("layers with learnable parameters:")
        for layer_key, layer in self.paras_dict.items():
            for sublayer_key, sublayer_paras in layer.items():
                print(layer_key, "\n", sublayer_key)
                print(sublayer_paras["W"].data.shape)
                print(sublayer_paras["b"].data.shape)
                print()

    def get_parameters(self, layer_name=None):
        """
        From paras_dict (OrderedDict), return a list of Dict object for sublayers
        that have learnable parameters. Each Dict object is the parameters attribute
        of a certain sublayer, sublayer.parameters, i.e., a dict {"W": W, "b": b}.
        If layer_name is supplied, then return only a list at the sublayers that belong to a particular class.

        Args:
            layer_name (str): name for the sublayer class from where parameters are
            extracted.
            Examples: "linear", "batchnorm", "conv1d"

        Returns:
            list:  list of Dict objects.
                Specially, each element of the list is a Dict() object at a
                certain sublayer; the Dict() object has key-value pair, e.g.,
                dict {"W": W, "b": b}, value W or b is the corresponding obj-Parameter <nn.modules.parameters>.

        """
        model_paras_sublist = []
        for layer_key, layer in self.paras_dict.items():
            for sublayer_key, sublayer_paras in layer.items():
                if layer_name is None or layer_name in sublayer_key:
                    model_paras_sublist.append(sublayer_paras)
                    print(layer_key, "\n", sublayer_key)
                    print(sublayer_paras["W"].data.shape)
                    print(sublayer_paras["b"].data.shape)
                    print()
        return model_paras_sublist

    def forward(self, x):
        """
        Args:
            x (np.array): (batch_size, num_input_channels, input_width)
        Returns:
            out (np.array): (batch_size, num_linear_neurons). logits.
        """

        # Iterate through each layer
        # Save output (necessary for error and loss)
        # self.output = x

        input = x
        for i in range(self.num_conv_layers):
            z = self.convolutional_layers[i].forward(input)
            input = self.activations[i].forward(z)
            if len(self.pool_layers) != 0:
                input = self.pool_layers[i].forward(input)
        input = self.flatten.forward(input)
        # input shape  (batch_size, in_channel * in_width)
        self.output = self.linear_layer.forward(input)

        return self.output

    def backward(self, dLdout):
        """
        Args:
            dLdout <np.dnarray>: (batch size, output_size)
                gradient of loss wrt output of the model
                It is the returned value from criterion.backward().
        Returns:
            dLdout (np.array): (batch size, num_input_channels, input_width)
            it is the gradient wrt input x (i.e, wrt output at the layer 0)
        """

        # Iterate through each layer in reverse order

        dLdA = self.linear_layer.backward(dLdout)  # backprop on linear layer
        dLdA = self.flatten.backward(dLdA)  # backprop on flatten layer
        dLdZ = 0  # gradient of output in a neuro wrt its input
        for i in range(self.num_conv_layers - 1, -1, -1):
            # iterates over the values self.num_conv_layers - 1, self.num_conv_layers - 2, ..., 1, 0
            if len(self.pool_layers) != 0:
                dLdA = self.pool_layers[i].backward(dLdA)
            dLdZ = np.multiply(dLdA, self.activations[i].backward())
            dLdA = self.convolutional_layers[i].backward(dLdZ)

        # gradient wrt input x (i.e, wrt output at the layer 0)
        dLdout = dLdA

        return dLdout


class Lenet5(object):
    """
    Lenet 5 (2D CNN)

    x (batch_size, in_channels=1, in_height=32, in_width=32)
    -> Conv2d (out_channels 6, kernel 5, stride 1)-> tanh ->maxpool (kernel 2)
    -> Conv2d (out_channels 16, kernel 5, stride 1)-> tanh ->maxpool (kernel 2)
    -> Flatten2D
    -> Linear (in_features=16*5*5, out_features=120) -> tanh
    -> Linear (in_features=120, out_features=84) -> tanh
    -> Linear (in_features=84, out_features=10) (-->identity)

    nlayers = 2+ 1+ 3=6

    We uses CrossEntropyLoss, thus the output is logits.

    Args:
        input_dims            (List/int)    : The height and width of the input to the first convolutional layer
        num_input_channels    (int)    : Number of channels for the input layer
        num_channels          (list/int)  : List containing number of (output) channels for each conv layer
        kernel_sizes          (list/int)  : List containing kernel width for each conv layer (kernel_size same for h, w)
        strides               (list/int)  : List containing stride size for each conv layer
        num_linear_neurons    (list/int)  : List containing Number of neurons (outputs) for each linear layer
        conv_activations      (list/obj-Activation <nn.modules.activation>)  : List of objects corresponding to the activation fn for each conv layer
        linear_activations    (list/obj-Activation <nn.modules.activation>)  : List of objects corresponding to the activation fn for each linear layer
        conv_weight_init_fn   (fn)     : Function to init each conv layers weights
        bias_init_fn          (fn)     : Function to initialize each conv layers AND the linear layers bias to 0
        linear_weight_init_fn (fn)     : Function to initialize the linear layers weights
        pool_kernel_sizes     (list/int)  : List containing kernel width for each pooling layer
        pool_mode             (str)    : "max" or "average" (default "max")

        note: if pooling layer is used, we require it to be applied after the activation of each conv layer,
        i.e., len(pool_kernel_sizes)=len(num_channels)
        and we use default stride for pooling layer, i.e., stride for pooling = kernel size,
        and pooling does not change channels numbers.
        Though not efficient, setting pool kernel size=1, is equivalent to not pooling at all.

    Attrs:
        train_mode: default=True
        num_conv_layers: number of conv layers
        num_linear_layers: number of linear layers
        num_pool_layers: number of pool layers
        conv_activations: a list of activations for conv layers
        linear_activations: a list of activations for linear layers
        flatten (fn):  Flatten2D()
        convolutional_layers (list/obj-Conv1D <nn.modules.conv>): a list of Conv1D objects
        linear_layers: (list/obj-Linear <nn.modules.linear>)
        pool_layers: (list/obj-Pooling1D <nn.modules.conv>)
        nlayers: number of total layers = number of conv layers + number of pool layers
                                + 1 (flatten)+ num of linear layers (including a linear output layer)
        layers_dict (OrderedDict of List):
            a ordered dict for layers: layers_dict

        paras_dict (Nested OrderedDict):
            dict for parameters of the model: paras_dict

    Methods:
        train
        eval
        forward
        backward
        print_structure
        get_parameters


    You can be sure that
    len(conv_activations) == len(num_channels) == len(kernel_sizes) == len(strides)
    """

    def __init__(
        self,
        input_dims,
        num_input_channels,
        num_channels,
        kernel_sizes,
        strides,
        num_linear_neurons,
        conv_activations,
        linear_activations,
        conv_weight_init_fn,
        bias_init_fn,
        linear_weight_init_fn,
        pool_kernel_sizes=None,
        pool_mode="max",
    ):
        self.train_mode = True
        self.num_conv_layers = len(num_channels)
        self.num_linear_layers = len(num_linear_neurons)
        if pool_kernel_sizes is not None:
            self.num_pool_layers = len(pool_kernel_sizes)
        self.conv_activations = conv_activations
        self.linear_activations = linear_activations
        self.flatten = None
        self.linear_layers = None
        self.nlayers = (
            self.num_conv_layers + self.num_pool_layers + self.num_linear_layers + 1
        )

        outChannel = num_input_channels
        output_h, output_w = [0, 0]
        input_h, input_w = input_dims

        self.layers_dict = OrderedDict()
        self.paras_dict = OrderedDict()

        self.convolutional_layers = []
        self.pool_layers = []
        for i in range(self.num_conv_layers):
            self.convolutional_layers.append(
                Conv2D(
                    in_channel=outChannel,
                    out_channel=num_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    weight_init_fn=conv_weight_init_fn,
                    bias_init_fn=bias_init_fn,
                )
            )
            outChannel = num_channels[i]
            output_h = int((input_h - kernel_sizes[i]) / strides[i]) + 1
            output_w = int((input_w - kernel_sizes[i]) / strides[i]) + 1

            input_h = output_h
            input_w = output_w
            # if there is pooling layers
            if pool_kernel_sizes is not None:
                self.pool_layers.append(
                    Pool2D(pool_size=pool_kernel_sizes[i], mode=pool_mode)
                )
                output_h = (input_h - pool_kernel_sizes[i]) // pool_kernel_sizes[i] + 1
                output_w = (input_w - pool_kernel_sizes[i]) // pool_kernel_sizes[i] + 1
                input_h = output_h
                input_w = output_w

        self.flatten = Flatten2D()

        inputSize = outChannel * input_h * input_w  # input size in the linear layers

        self.linear_layers = []
        for i in range(self.num_linear_layers):
            self.linear_layers.append(
                Linear(
                    in_features=inputSize,
                    out_features=num_linear_neurons[i],
                    weight_init_fn=linear_weight_init_fn,
                    bias_init_fn=bias_init_fn,
                )
            )
            inputSize = num_linear_neurons[i]

        # create a ordered dict for layers: layers_dict

        idx_conv = 0
        idx_pool = 0
        for i in range(self.nlayers - (1 + self.num_linear_layers)):
            self.layers_dict["layer" + str(i)] = []
            if i % 2 == 0:
                self.layers_dict["layer" + str(i)].append(
                    self.convolutional_layers[idx_conv]
                )
                self.layers_dict["layer" + str(i)].append(
                    self.conv_activations[idx_conv]
                )
                idx_conv += 1
            if (len(self.pool_layers) != 0) & (i % 2 != 0):
                self.layers_dict["layer" + str(i)].append(self.pool_layers[idx_pool])
                idx_pool += 1

        i += 1
        self.layers_dict["layer" + str(i)] = []
        self.layers_dict["layer" + str(i)].append(self.flatten)
        i += 1

        for j in range(self.num_linear_layers):
            self.layers_dict["layer" + str(j + i)] = []
            self.layers_dict["layer" + str(j + i)].append(self.linear_layers[j])
            self.layers_dict["layer" + str(j + i)].append(self.linear_activations[j])

        ## construct dict for parameters of the model: "paras_dict"

        for i in range(self.nlayers):
            layer = self.layers_dict["layer" + str(i)]
            self.paras_dict["layer" + str(i)] = {}
            sublayer_idx = 0
            for sublayer in layer:
                name = "(" + str(sublayer_idx) + ")"
                if isinstance(sublayer, Conv2D):
                    self.paras_dict["layer" + str(i)][
                        name + "conv1d"
                    ] = sublayer.parameters
                if isinstance(sublayer, Linear):
                    self.paras_dict["layer" + str(i)][
                        name + "linear"
                    ] = sublayer.parameters
                sublayer_idx += 1

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        inner_lines = "\n".join("%s:%s" % (k, v) for k, v in self.layers_dict.items())
        return inner_lines

    def train(self):
        # Do not modify this method
        self.train_mode = True

    def eval(self):
        # Do not modify this method
        self.train_mode = False

    def print_structure(self):
        """
        Print the structure of layers_dict and paras_dict
        """
        print("-----------------------")
        print("The model architecture:")
        for layer_key, layer_value in self.layers_dict.items():
            # Print the current layer name
            print(f"{layer_key}:")
            # Iterate through the sublayers and print them with indentation
            for i, element in enumerate(layer_value):
                sublayer_name = f"sublayer{i}"
                print(f"\t{sublayer_name}: {element}")
        print()
        print("---------------------------------")
        print("layers with learnable parameters:")
        for layer_key, layer in self.paras_dict.items():
            for sublayer_key, sublayer_paras in layer.items():
                print(layer_key, "\n", sublayer_key)
                print(sublayer_paras["W"].data.shape)
                print(sublayer_paras["b"].data.shape)
                print()

    def get_parameters(self, layer_name=None):
        """
        From paras_dict (OrderedDict), return a list of Dict object for sublayers
        that have learnable parameters. Each Dict object is the parameters attribute
        of a certain sublayer, sublayer.parameters, i.e., a dict {"W": W, "b": b}.
        If layer_name is supplied, then return only a list at the sublayers that belong to a particular class.

        Args:
            layer_name (str): name for the sublayer class from where parameters are
            extracted.
            Examples: "linear", "batchnorm", "conv1d"

        Returns:
            list:  list of Dict objects.
                Specially, each element of the list is a Dict() object at a
                certain sublayer; the Dict() object has key-value pair, e.g.,
                dict {"W": W, "b": b}, value W or b is the corresponding obj-Parameter <nn.modules.parameters>.

        """
        model_paras_sublist = []
        for layer_key, layer in self.paras_dict.items():
            for sublayer_key, sublayer_paras in layer.items():
                if layer_name is None or layer_name in sublayer_key:
                    model_paras_sublist.append(sublayer_paras)
                    print(layer_key, "\n", sublayer_key)
                    print(sublayer_paras["W"].data.shape)
                    print(sublayer_paras["b"].data.shape)
                    print()
        return model_paras_sublist

    def forward(self, x):
        """
        Args:
            x (np.array): (batch_size, input_channels, in_height, in_width)
        Returns:
            out (np.array): (batch_size, num_linear_neurons). logits.
        """

        # Iterate through each layer
        # Save output (necessary for error and loss)
        # self.output = x

        output = x

        for i in range(self.nlayers):
            layer = self.layers_dict["layer" + str(i)]
            for sublayer in layer:
                if isinstance(sublayer, Dropout):
                    output = sublayer.forward(output, train=self.train_mode)
                elif isinstance(sublayer, Linear):
                    output = sublayer.forward(output)
                elif isinstance(sublayer, BatchNorm1d):
                    output = sublayer.forward(output, train=self.train_mode)
                elif isinstance(sublayer, Conv2D):
                    output = sublayer.forward(output)
                elif isinstance(sublayer, Activation):
                    output = sublayer.forward(output)
                elif isinstance(sublayer, Pool2D):
                    output = sublayer.forward(output)
                elif isinstance(sublayer, Flatten2D):
                    output = sublayer.forward(output)
                else:
                    raise NotImplementedError

        self.output = output

        return self.output

    def backward(self, dLdout):
        """
        Args:
            dLdout <np.dnarray>: (batch size, output_size)
                gradient of loss wrt output of the model
                It is the returned value from criterion.backward().
        Returns:
            dLdout (np.array): (batch size, num_input_channels, input_width)
            it is the gradient wrt input x (i.e, wrt output at the layer 0)
        """

        # Iterate through each layer in reverse order

        # backward() perform backpropagation through the model
        # populating gradient of all Parameter <nn.modules.parameters>

        # Hidden layers
        d_in_dict = OrderedDict()
        # a dict of gradients of loss wrt input (input of neurons)
        # the list stores dLdin in backward order, length = self.nlayers

        d_out_dict = OrderedDict()
        # a dict of gradients of loss wrt outout (output of neurons)
        # the list stores dLdout in backward order, length = self.nlayers + 1
        # i.e., including the dLdx, where x is the input variables to
        # the network (equiv. the output of the 0 layer)

        for i in range(self.nlayers):
            d_in_dict["layer" + str(i)] = None
            d_out_dict["layer" + str(i)] = None

        d_out_dict["layer" + str(self.nlayers)] = dLdout  # loss wrt the output layer
        # d_out_dict['layer0'] is loss wrt the input laye

        for i in reversed(range(self.nlayers)):
            # iterate from self.nlayers - 1, self.nlayers - 2, ..., 1, 0
            layer = self.layers_dict["layer" + str(i)]
            for sublayer in reversed(layer):
                # the following will compute d_in
                if isinstance(sublayer, Activation):
                    # this a sublayer of activation
                    d_act = sublayer.backward()  # grad of d_out wrt d_in
                    dLdin = np.multiply(d_out_dict["layer" + str(i + 1)], d_act)
                    d_in_dict["layer" + str(i)] = dLdin
                # the following will compute d_out
                elif isinstance(sublayer, Linear):
                    # invoke Linear backward
                    dLdA = sublayer.backward(d_in_dict["layer" + str(i)])
                    d_out_dict["layer" + str(i)] = dLdA
                elif isinstance(sublayer, Pool2D):
                    dLdA = sublayer.backward(d_out_dict["layer" + str(i + 1)])
                    d_out_dict["layer" + str(i)] = dLdA
                elif isinstance(sublayer, Conv2D):
                    dLdA = sublayer.backward(d_in_dict["layer" + str(i)])
                    d_out_dict["layer" + str(i)] = dLdA
                elif isinstance(sublayer, Flatten2D):
                    dLdA = sublayer.backward(d_out_dict["layer" + str(i + 1)])
                    d_out_dict["layer" + str(i)] = dLdA
                else:
                    raise NotImplementedError

        return d_in_dict, d_out_dict


###############################################################################
########################## begin of playground ################################
###############################################################################


###############################################################################
########################## end of playground ################################
###############################################################################
