import numpy as np
import os
import sys

sys.path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nn.modules.loss import *
from nn.modules.activation import *
from nn.modules.batchnorm import *
from nn.modules.linear import *
from nn.modules.dropout import *
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


class MLP(object):
    """
    A simple multilayer perceptron

    Args:
        input_size  (int): _description_
        output_size (int): _description_
        hiddens     (list/int):  a list of sizes of dimensions of hidden layers
                        (excluding input size, and output size)
        activations (list/obj-Activation <nn.modules.activation>):
            a list of activation functions (first hidden layer to output layer)
            In geenral, use identity activation for the output layer,
            as we use CrossEntropyLoss that takes in logits.
        weight_init_fn (fn): initializer for W
        bias_init_fn   (fn): initializer for b
        num_bn_layers  (int):number of bn layers, applied only to the first
                        num_bn_layers layers.
        drop_p         (float): dropout probability. Defaults to 0.0.


    Attrs:
        train_mode: if the model is in training mode
        num_bn_layers: number of bn layers
        bn: if (num_bn_layers > 0)
        drop_p: dropout probability
        drop: if (drop_p > 0.0)
        nlayers: number of hidden layers + 1 output layer
                i.e, total number of layers with learnable params
        layersDim: a list of dimensions from input to output layer
        input_size: input_size
        output_size: output_size
        activations: a list of Activation <nn.modules.activation>
        linear_layers: a list of Linear <nn.modules.linear> objects
        bn_layers: a list of BatchNorm1D <nn.modules.batchnorm> objects
        layers_dict (OrderedDict): a ordered dict for layers: layers_dict

        paras_dict (OrderedDict): dict for parameters of the model: paras_dict

    Methods:
        train
        eval
        print_structure
        get_parameters
        forward
        backward

    """

    def __init__(
        self,
        input_size,
        output_size,
        hiddens,
        activations,
        weight_init_fn,
        bias_init_fn,
        num_bn_layers=0,
        drop_p=0.0,
    ):
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.drop_p = drop_p
        self.drop = drop_p > 0.0
        self.nlayers = (
            len(hiddens) + 1
        )  # number of hidden layers + 1 output layer; total number of layers with params
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations

        self.layers_dict = OrderedDict()
        self.paras_dict = OrderedDict()
        self.layersDim = [input_size] + hiddens + [output_size]
        # a list of sizes of dims from input to output layer

        # instantiate linear layers
        self.linear_layers = [
            Linear(inSize, outSize, weight_init_fn, bias_init_fn)
            for inSize, outSize in zip(self.layersDim[:-1], self.layersDim[1:])
        ]

        # instantiate bn layers
        if self.bn:
            self.bn_layers = (
                []
            )  # store a list of BatchNorm1D <nn.modules.batchnorm> objects
            for i in range(self.num_bn_layers):
                self.bn_layers.append(
                    BatchNorm1d(self.layersDim[i + 1], weight_init_fn, bias_init_fn)
                )

        # create a ordered dict for layers: "layers_dict"


        # note: dropout is not applied to the output layer, but to input layer
        for i in range(self.nlayers):
            self.layers_dict["layer" + str(i)] = []
            if (self.drop is True) & (i != self.nlayers - 1):
                self.layers_dict["layer" + str(i)].append(Dropout(self.drop_p))
            self.layers_dict["layer" + str(i)].append(self.linear_layers[i])
            if i < self.num_bn_layers:
                self.layers_dict["layer" + str(i)].append(self.bn_layers[i])
            self.layers_dict["layer" + str(i)].append(self.activations[i])

        ## construct dict for parameters of the model: "paras_dict"

        for i in range(self.nlayers):
            layer = self.layers_dict["layer" + str(i)]
            self.paras_dict["layer" + str(i)] = {}
            sublayer_idx = 0
            for sublayer in layer:
                name = "(" + str(sublayer_idx) + ")"
                if isinstance(sublayer, Linear):
                    self.paras_dict["layer" + str(i)][
                        name + "linear"
                    ] = sublayer.parameters
                if isinstance(sublayer, BatchNorm1d):
                    self.paras_dict["layer" + str(i)][
                        name + "batchnorm"
                    ] = sublayer.parameters
                sublayer_idx += 1

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        inner_lines = "\n".join("%s:%s" % (k, v) for k, v in self.layers_dict.items())
        return inner_lines

    def train(self):
        self.train_mode = True

    def eval(self):
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
            x (np.array): (batch size, input_size)

        Returns:
            out (np.array): (batch size, output_size)
            output of the model (to be fed into loss function)
        """
        # Complete the forward pass through your entire MLP.
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
                elif isinstance(sublayer, Activation):
                    output = sublayer.forward(output)
                else:
                    raise NotImplementedError

        return output

    def backward(self, dLdout):
        """
        Args:
            dLdout <np.dnarray>: (batch size, output_size)
                gradient of loss wrt output of the model
                It is the returned value from criterion.backward().

        Returns:
            d_in_dict
            d_out_dict
            d_out_bm_dict

        """

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

        d_out_bm_dict = OrderedDict()
        # a dict of gradients of loss wrt outout (output of BM sublayer)
        # the list stores dLdout in backward order

        for i in range(self.nlayers):
            d_in_dict["layer" + str(i)] = None
            d_out_bm_dict["layer" + str(i)] = None
            d_out_dict["layer" + str(i)] = None

        d_out_dict["layer" + str(self.nlayers)] = dLdout  # loss wrt the output layer
        # d_out_dict['layer0'] is loss wrt the input layer

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
                elif isinstance(sublayer, BatchNorm1d):
                    dLdBZ = d_in_dict["layer" + str(i)]
                    d_out_bm_dict["layer" + str(i)] = dLdBZ
                    # invoke batchnorm backward
                    dLdin = sublayer.backward(dLdBZ)
                    d_in_dict.update({"layer" + str(i): dLdin})

                # the following will compute d_out
                elif isinstance(sublayer, Linear):
                    # invoke Linear backward
                    dLdA = sublayer.backward(d_in_dict["layer" + str(i)])
                    d_out_dict["layer" + str(i)] = dLdA
                elif isinstance(sublayer, Dropout):
                    # invoke Dropout backward
                    dLdA = sublayer.backward(d_out_dict["layer" + str(i)])
                    d_out_dict.update({"layer" + str(i): dLdA})
                else:
                    raise NotImplementedError

        return d_in_dict, d_out_dict, d_out_bm_dict


"""
###############################################################################
########################## begin of playground ################################
###############################################################################

# mlp = nn.MLP(784, 10, [32, 32, 32], [nn.Sigmoid(), nn.Sigmoid(), nn.Sigmoid(), nn.Identity()],
#                 np.random.randn, bias_init,
#                 nn.SoftmaxCrossEntropy(),
#                 1e-3, momentum=0.856)
# visualize_training_statistics(mlp, dset, epochs, batch_size, savepath)

layersDim = [784] + [32, 32, 32] + [10]

list(zip(layersDim[:-1], layersDim[1:]))


x = np.random.randn(3, 2)
bias_init(x)
weight_init_randn(x)
np.random.randn(*x.shape)

linear_eg = Linear(10, 10, weight_init_LeCun, bias_init)
linear_eg.W
linear_eg.b
print(repr(linear_eg.W))

"(" + ", ".join([str(s) for s in linear_eg.W.data.shape]) + ")"

batch_eg = BatchNorm1d(5, weight_init_fn=weight_init_randn, bias_init_fn=bias_init)
batch_eg.BW
batch_eg.Bb
batch_eg.parameters

hasattr(linear_eg, "b")


layersDim = [784] + [32, 32, 32] + [10]
linear_layers = [
    Linear(inSize, outSize, weight_init_LeCun, bias_init)
    for inSize, outSize in zip(layersDim[:-1], layersDim[1:])
]

nlayers = len(layersDim) - 1
# If batch norm, add batch norm layers into the list 'self.bn_layers'
# recall batchnorm is applied before activation

bn_layers = []  # store a list of nn.modules.BatchNorm objects
num_bn_layers = 1
for i in range(num_bn_layers):
    bn_layers.append(
        BatchNorm1d(
            layersDim[i + 1], weight_init_fn=weight_init_randn, bias_init_fn=bias_init
        )
    )

activations = [Sigmoid(), Sigmoid(), Sigmoid(), Identity()]

drop = True
drop_p = 0.5


layers_dict = OrderedDict()
# create the layers_dict
# note the order of sublayers:
# dropout -> linear -> bm -> activations
# dropout is not applied to the output layer, but to input layer
for i in range(nlayers):
    layers_dict["layer" + str(i)] = []
    if (drop == True) & (i != nlayers - 1):
        layers_dict["layer" + str(i)].append(Dropout(drop_p))
    layers_dict["layer" + str(i)].append(linear_layers[i])
    if i < num_bn_layers:
        layers_dict["layer" + str(i)].append(bn_layers[i])
    layers_dict["layer" + str(i)].append(activations[i])


print_keys(layers_dict)

layers_dict

for keys, values in layers_dict.items():
    print(keys)
    print(values)


layers_dict["layer0"][1].parameters.keys()
type(layers_dict["layer0"][1].parameters["W"])

linear_layers[0].parameters
linear_layers[0].parameters.keys()
type(linear_layers[0].parameters["W"])

bn_layers[0].parameters.keys()
bn_layers[0].BW
bn_layers[0].parameters["W"]


### Create parameters dict of the model

paras_dict = OrderedDict()
# a nested dictionary
# outermost dictionary:
# key: layer index
# value: inner dictionary whose key-value is sublayer index and the parameters

## construct dict for parameters
for i in range(nlayers):
    layer = layers_dict["layer" + str(i)]
    paras_dict["layer" + str(i)] = {}
    sublayer_idx = 0
    for sublayer in layer:
        name = "(" + str(sublayer_idx) + ")"
        if isinstance(sublayer, Linear):
            paras_dict["layer" + str(i)][name + "linear"] = sublayer.parameters
        if isinstance(sublayer, BatchNorm1d):
            paras_dict["layer" + str(i)][name + "batchnorm"] = sublayer.parameters
        sublayer_idx += 1


layers_dict.keys()
paras_dict
len(paras_dict)
len(paras_dict["layer0"])
paras_dict["layer1"].keys()
print_keys(paras_dict)


####

x = np.random.randn(10, 784)

for i in layers_dict["layer0"]:
    print(i)

isinstance(layers_dict["layer0"][0], Dropout)
isinstance(layers_dict["layer0"][2], BatchNorm1d)

train_mode = True

def forward(x):
    # Complete the forward pass through your entire MLP.
    input = x

    for i in range(nlayers):
        layer = layers_dict["layer" + str(i)]
        for sublayer in layer:
            if isinstance(sublayer, Dropout):
                input = sublayer.forward(input, train=train_mode)
            elif isinstance(sublayer, Linear):
                input = sublayer.forward(input)
            elif isinstance(sublayer, BatchNorm1d):
                input = sublayer.forward(input, train=train_mode)
            elif isinstance(sublayer, Activation):
                input = sublayer.forward(input)
            else:
                print("Oops! There is one sublayer not recognized.")

    return input


forward(x).shape

targets = np.array([7, 2, 1, 0, 4, 1, 4, 9, 5, 9])


targets = data_processor.to_onehot(targets, num_classes=10)
targets

####  backward

nlayers

criterion = CrossEntropyLoss()

loss = criterion.forward(layers_dict["layer" + str(nlayers - 1)][-1].A, targets)
dLdA = criterion.backward()  # gradient of loss wrt output

# Hidden layers
d_in_dict = OrderedDict()
# a dict of gradients of loss wrt input (input of neurons)
# the list stores dLdin in backward order, length = self.nlayers

d_out_dict = OrderedDict()
# a dict of gradients of loss wrt outout (output of neurons)
# the list stores dLdout in backward order, length = self.nlayers + 1
# i.e., including the dLdx, where x is the input variables to
# the network

d_out_bm_dict = OrderedDict()
# a dict of gradients of loss wrt outout (output of BM sublayer)
# the list stores dLdout in backward order

for i in range(nlayers):
    d_in_dict["layer" + str(i)] = None
    d_out_bm_dict["layer" + str(i)] = None
    d_out_dict["layer" + str(i)] = None

d_out_dict["layer" + str(nlayers)] = None  # loss wrt the output layer
# d_out_dict['layer0'] is loss wrt the input layer


# layers_dict['layer'+str(nlayers-1)][-1] is the activation in the output layer
last_sublayer = layers_dict["layer" + str(nlayers - 1)]
loss = criterion.forward(last_sublayer[-1].A, targets)
dLdout = criterion.backward()  # gradient of loss wrt output
d_out_dict["layer" + str(nlayers)] = dLdout


for i in reversed(range(nlayers)):
    layer = layers_dict["layer" + str(i)]
    for sublayer in reversed(layer):

        # the following will compute d_in
        if isinstance(sublayer, Activation):
            # this a sublayer of activation
            d_act = sublayer.backward()  # grad of d_out wrt d_in
            dLdin = np.multiply(d_out_dict["layer" + str(i + 1)], d_act)
            d_in_dict["layer" + str(i)] = dLdin
        elif isinstance(sublayer, BatchNorm1d):
            dLdBZ = d_in_dict["layer" + str(i)]
            d_out_bm_dict["layer" + str(i)] = dLdBZ
            # invoke batchnorm backward
            dLdin = sublayer.backward(dLdBZ)
            d_in_dict.update({"layer" + str(i): dLdin})

        # the following will compute d_out
        elif isinstance(sublayer, Linear):
            # invoke Linear backward
            dLdA = sublayer.backward(d_in_dict["layer" + str(i)])
            d_out_dict["layer" + str(i)] = dLdA
        elif isinstance(sublayer, Dropout):
            # invoke Dropout backward
            dLdA = sublayer.backward(d_out_dict["layer" + str(i)])
            d_out_dict.update({"layer" + str(i): dLdA})
        else:
            print("Oops! There is one sublayer not recognized.")


len(d_in_dict), len(d_out_bm_dict), len(d_out_dict)

layers_dict["layer0"][2].BW.data
layers_dict["layer0"][2].parameters["W"].data

#### work with optimizer.step()


####   SGD_decay

# SgdDecay

layers_dict
paras_dict
paras_dict.keys()


print_keys(layers_dict)
print_keys(paras_dict)


## create a complete parameters list
## for feeding into optimizers internally
## The list is alias of paras_dict that share same memory
## of parameter.data and parameter.grad.
model_paras_list = []
for layer_key, layer in paras_dict.items():
    for sublayer_key, sublayer_paras in layer.items():
        print(layer_key, "\n", sublayer_key)
        print(sublayer_paras["W"].data.shape)
        print(sublayer_paras["b"].data.shape)
        print()
        model_paras_list.append(sublayer_paras)

len(model_paras_list)
model_paras_list[0]["W"].data
model_paras_list[1]["W"].data.shape


for layer_key, layer in paras_dict.items():
    for sublayer_key, sublayer_paras in layer.items():
        print(layer_key, "\n", sublayer_key)
        print(sublayer_paras["W"].data.shape)
        print(sublayer_paras["b"].data.shape)
        print()


## create separate parameters list
## for different type of sublayers
## for feeding into optimizers internally
## The list is alias of paras_dict that share same memory
## of Parameter.data and Parameter.grad.


def get_parameters(paras_dict, layer_name=None):
    model_paras_sublist = []
    for layer_key, layer in paras_dict.items():
        for sublayer_key, sublayer_paras in layer.items():
            if layer_name is None or layer_name in sublayer_key:
                model_paras_sublist.append(sublayer_paras)
                print(layer_key, "\n", sublayer_key)
                print(sublayer_paras["W"].data.shape)
                print(sublayer_paras["b"].data.shape)
                print()
    return model_paras_sublist


paras_list = get_parameters(paras_dict)
linear_paras_list = get_parameters(paras_dict, layer_name="linear")
bn_paras_list = get_parameters(paras_dict, layer_name="batchnorm")

len(linear_paras_list)
linear_paras_list[0]["W"].data
linear_paras_list[0]["W"].grad
len(linear_paras_list)
len(bn_paras_list)
bn_paras_list[0]["W"].data
# should be same to
paras_dict["layer0"]["(2)batchnorm"]["W"].data

##

optimizer_linear = SgdDecay(linear_paras_list, lr=0.1, momentum=0.856, l2_weight=0.0)
optimizer_bm = SgdDecay(bn_paras_list, lr=0.1, momentum=0, l2_weight=0.0)

linear_paras_list[0]["W"].data
optimizer_linear.step()
linear_paras_list[0]["W"].data

linear_paras_list[0]["W"].grad
optimizer_linear.zero_grad()

bn_paras_list[0]["W"].data
optimizer_bm.step()
bn_paras_list[0]["W"].data

###############################################################################
######################  end of playground ###################################
###############################################################################
"""
