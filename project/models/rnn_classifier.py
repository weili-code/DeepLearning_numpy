import numpy as np
import sys

sys.path.append("project")
from nn.modules.rnn_cell import *
from nn.modules.linear import *
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


class RNNClassifier(object):
    """RNN Classifier class.
    Many-to-one RNN. Same RNN Cell is used across time at each layer.
    The Linear output layer is only at the last time stamp.
    There is not activation after the output layer (or equivalently, identity activation).

                          (output :logits)
    layer 2                    Linear
    layer 1    RNN  --- RNN --- RNN
    layer 0    RNN  --- RNN --- RNN
               time0---time1---time2


    Args:
        input_size (int): The dimensionality of the input vectors.
        hidden_size (int): The dimensionality of the hidden state vectors in each layer.
        output_size (int): The dimensionality of the output vectors.
        num_layers (int, optional): The number of hidden layers in the RNN. Defaults to 2.

        note: the time length is determined by the input used in forward()

    Attributes:
        rnn (list): A list containing the RNN layers that make up the hidden layers.
                    len(rnn)=num_layers
        output_layer (Linear): A linear layer that maps the last hidden state to the output.
        hiddens (list): A list that stores the hidden states at each time step.
                        a list of length (seq_len+1), each of shape (num_layers, batch_size, hidden_size)
                        a total of seq_len+1 time stamps (including time 0, initial time)

    Methods:
        init_weights: Initialize the weight matrices and bias vectors for the RNN and output layer.
        forward: Forward pass for the RNN over multiple time steps and layers.
        backward: Backward pass for backpropagation through time (BPTT).

    Note that: for the forward(), the input is x (np.array) of shape (batch_size, seq_len, input_size).

    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers  # number of hidden layers

        self.rnn = [
            RNNCell(input_size, hidden_size)
            if i == 0
            else RNNCell(hidden_size, hidden_size)
            for i in range(num_layers)
        ]
        # note that self.rnn contains as many RNN cells as number of layers
        # in the backprop, these cells objects will be re-used across time (going from
        # last time to initial time).

        self.output_layer = Linear(hidden_size, output_size)

        # store hidden states at each time step, a total of seq_len+1 time stamps (including time 0, initial time)
        # hidden states at each time is of (num_layers, batch_size, hidden_size)
        # self.hiddens: a list of length (seq_len+1), each of shape (num_layers, batch_size, hidden_size)
        self.hiddens = []
        self.train_mode = True

        ##------- define layers_dict and paras_dict --------##
        self.layers_dict = OrderedDict()
        self.paras_dict = OrderedDict()
        # create a ordered dict for layers: "layers_dict"


        for i in range(self.num_layers + 1):
            self.layers_dict["layer" + str(i)] = []
            if i == self.num_layers:
                self.layers_dict["layer" + str(self.num_layers)].append(
                    self.output_layer
                )
            else:
                self.layers_dict["layer" + str(i)].append(self.rnn[i])

        ## construct dict for parameters of the model: "paras_dict"

        for i in range(self.num_layers + 1):
            layer = self.layers_dict["layer" + str(i)]
            self.paras_dict["layer" + str(i)] = {}
            sublayer_idx = 0
            for sublayer in layer:
                name = "(" + str(sublayer_idx) + ")"
                if isinstance(sublayer, Linear):
                    self.paras_dict["layer" + str(i)][
                        name + "linear"
                    ] = sublayer.parameters
                if isinstance(sublayer, RNNCell):
                    self.paras_dict["layer" + str(i)][
                        name + "RNNCell"
                    ] = sublayer.parameters
                sublayer_idx += 1

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

    def init_weights(self, rnn_weights, linear_weights):
        """Initialize weights.
        Initialize the weight matrices and bias vectors for both the RNN and the output layer using numpy arrays.

        An alternative is to initialize the paramters returned from get_parameters().

        Args:
            rnn_weights: A list of lists containing the weight matrices and bias vectors values for each RNN cell.
                        [
                            [W_ih_l0, W_hh_l0, b_ih_l0, b_hh_l0],
                            [W_ih_l1, W_hh_l1, b_ih_l1, b_hh_l1],
                            ...
                        ]

            linear_weights: [W, b]
                        A list containing the weight matrix and bias vector values.
                        Here b is 1D array but will be converted to (out, 1).

        """
        for i, rnn_cell in enumerate(self.rnn):
            rnn_cell.init_weights(*rnn_weights[i])
            # *rnn_weights[i] unpack rnn_weights[i]
        self.output_layer.W.data = linear_weights[0]
        self.output_layer.b.data = linear_weights[1].reshape(-1, 1)

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
                if layer_key != "layer" + str(self.num_layers):
                    # layers except the last one are all RNN.
                    print(sublayer_paras["W_ih"].data.shape)
                    print(sublayer_paras["W_hh"].data.shape)
                    print(sublayer_paras["b_ih"].data.shape)
                    print(sublayer_paras["b_hh"].data.shape)
                else:  # the last layer is linear layer
                    print(sublayer_paras["W"].data.shape)
                    print(sublayer_paras["b"].data.shape)
                print()

    def get_parameters(self, layer_name=None):
        """
        From paras_dict (OrderedDict), return a list of Dict object for sublayers
        that have learnable parameters. Each Dict object is the parameters attribute
        of a certain sublayer, sublayer.parameters, i.e., a dict {"W_ih": W_ih, "W_hh": W_hh,  "b_ih": b_ih, "b_hh": b_hh}
        If layer_name is supplied, then return only a list at the sublayers that belong to a particular class.

        Args:
            layer_name (str): name for the sublayer class from where parameters are
            extracted.
            Examples: "linear", "rnn"

        Returns:
            list:  list of Dict objects.
                Specially, each element of the list is a Dict() object at a
                certain sublayer; the Dict() object has key-value pair, e.g.,
                dict {"W_ih": W_ih, "W_hh": W_hh,  "b_ih": b_ih, "b_hh": b_hh}, value W or b is the corresponding obj-Parameter <nn.modules.parameters>.
        """
        model_paras_sublist = []
        for layer_key, layer in self.paras_dict.items():
            for sublayer_key, sublayer_paras in layer.items():
                if layer_name is None or layer_name in sublayer_key:
                    model_paras_sublist.append(sublayer_paras)
                    print(layer_key, "\n", sublayer_key)
                    if layer_key != "layer" + str(self.num_layers):
                        # layers except the last one are all RNN.
                        print(sublayer_paras["W_ih"].data.shape)
                        print(sublayer_paras["W_hh"].data.shape)
                        print(sublayer_paras["b_ih"].data.shape)
                        print(sublayer_paras["b_hh"].data.shape)
                    else:
                        # last layer is linear layer
                        print(sublayer_paras["W"].data.shape)
                        print(sublayer_paras["b"].data.shape)
                    print()
        return model_paras_sublist

    def __call__(self, x, h_0=None):
        return self.forward(x, h_0)

    def forward(self, x, h_0=None):
        """RNN classifier forward, multiple layers, multiple time steps.

        Args:
            x (np.array): The input sequences, of shape (batch_size, seq_len, input_size).
            h_0 (np.array, optional): The initial hidden states, of shape (num_layers, batch_size, hidden_size).
                                      Defaults to zeros if not specified.

        Returns:
            logits (np.array): The output logits, of shape (batch_size, output_size).

        """
        self.hiddens = []  # initialize the hiddens list

        # Get the batch size and sequence length, and initialize the hidden
        # vectors given the paramters.
        batch_size, seq_len = x.shape[0], x.shape[1]
        if h_0 is None:
            hidden = np.zeros(
                (self.num_layers, batch_size, self.hidden_size), dtype=float
            )
        else:
            hidden = h_0

        # Save x and append the hidden vector to the hiddens list
        self.x = x
        self.hiddens.append(hidden.copy())
        logits = None

        ### code
        # Iterate through the sequence
        #   Iterate over the length of your self.rnn (through the layers)
        #       Run the rnn cell with the correct parameters and update
        #       the parameters as needed. Update hidden.
        #   Similar to above, append a copy of the current hidden array to the hiddens list

        # Iterate through the sequence
        for t in range(seq_len):
            xInput = self.x[:, t, :]  # (batch_size, input_size)
            hidden = []  # a list to store hidden states at the given time
            # Iterate over the length of your self.rnn (through the layers)
            for i in range(len(self.rnn)):
                # Run the rnn cell with the correct parameters and update
                # the parameters as needed. Update hidden.
                h_ti = self.rnn[i].forward(xInput, self.hiddens[-1][i])
                # self.hiddens[-1][i]： the i-th layer of hidden state in last time stamp
                # h_ti: (batch_size, hidden_size)
                xInput = h_ti
                hidden.append(h_ti)
            # Similar to above, append a copy of the current hidden array to the hiddens list
            # hidden is of shape (num_layers, batch_size, hidden_size)
            hidden = np.array(hidden)
            self.hiddens.append(hidden.copy())

        # Get the outputs from the last time step using the linear layer and return it
        logits = self.output_layer.forward(xInput)

        return logits

    def backward(self, delta):
        """RNN Back Propagation Through Time (BPTT).

        Here, we allow general batch size, and we define a single cell at one time stamp and one layer, for many-to-one RNN.

        In the backward method, we continuously update dh for the previous time until we have that for
        the initial hidden states. That is, we do not save all dh for all time and layers, but just return
        the dh for the initial hidden states.

        Args:
            delta (np.array): The gradient of the loss with respect to the output logits,
                              of shape (batch_size, output_size).
                            i.e., gradient w.r.t. the last time step output (of a linear layer).

        Returns:
            dh_0 (np.array): The gradient with respect to the initial hidden states,
                             of shape (num_layers, batch_size, hidden_size).

        """
        # Initilizations
        batch_size, seq_len = self.x.shape[0], self.x.shape[1]
        dh = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        dh[-1] = self.output_layer.backward(delta)


        # Iterate in reverse order of time (from seq_len-1 to 0)
        for t in range(seq_len - 1, -1, -1):
            # Iterate in reverse order of layers (from num_layers-1 to 0)
            for i in range(self.num_layers - 1, -1, -1):
                # Get h_prev_l either from hiddens or x depending on the layer
                #   (Recall that hiddens has an extra initial hidden state
                #       so, self.hiddens[t + 1] is the hidden state in time t)
                h_prev_l = self.hiddens[t + 1][i - 1] if i != 0 else self.x[:, t, :]
                h_prev_t = self.hiddens[t][i]

                # Use dh and hiddens to get the other parameters for the backward method
                #   (Recall that hiddens has an extra initial hidden state)
                dx_prev_l, dh_prev_t = self.rnn[i].backward(
                    dh[i], self.hiddens[t + 1][i], h_prev_l, h_prev_t
                )
                # the first argument in self.rnn[i].backward is the delta.

                # Update dh with the new dh from the backward pass of the rnn cell
                # note: no increment here
                dh[i] = dh_prev_t
                # If you aren't at the first layer, you will want to add dx to
                #   the gradient from l-1th layer, using += increment
                if i != 0:
                    dh[i - 1] += dx_prev_l

        # Normalize dh by batch_size since initial hidden states are also treated
        #   as parameters of the network (divide by batch size)
        return dh / batch_size


###############################################################
################# Playground #####################
###############################################################


# import torch
# import torch.nn

# np.random.seed(11785)
# torch.manual_seed(11785)

# rnn_layers = 2
# batch_size = 5
# seq_len = 10
# input_size = 40
# hidden_size = 32  # hidden_size > 100 will cause precision error
# output_size = 138

# data_x = np.random.randn(batch_size, seq_len, input_size)
# data_y = np.random.randint(0, output_size, batch_size)

# # My model
# my_rnn_model = RNNClassifier(
#     input_size, hidden_size, output_size, num_layers=rnn_layers
# )


# # Reference Pytorch RNN Model
# class ReferenceModel(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, rnn_layers=2):
#         super(ReferenceModel, self).__init__()
#         self.rnn = torch.nn.RNN(
#             input_size, hidden_size, num_layers=rnn_layers, bias=True, batch_first=True
#         )
#         self.output = torch.nn.Linear(hidden_size, output_size)

#     def forward(self, x, init_h=None):
#         out, hidden = self.rnn(x, init_h)
#         out = self.output(out[:, -1, :])
#         return out


# # Reference model
# rnn_model = ReferenceModel(input_size, hidden_size, output_size, rnn_layers=rnn_layers)
# model_state_dict = rnn_model.state_dict()

# rnn_weights = [
#     [
#         model_state_dict["rnn.weight_ih_l%d" % l].numpy(),
#         model_state_dict["rnn.weight_hh_l%d" % l].numpy(),
#         model_state_dict["rnn.bias_ih_l%d" % l].numpy(),
#         model_state_dict["rnn.bias_hh_l%d" % l].numpy(),
#     ]
#     for l in range(rnn_layers)
# ]

# fc_weights = [
#     model_state_dict["output.weight"].numpy(),
#     model_state_dict["output.bias"].numpy(),
# ]

# my_rnn_model.init_weights(rnn_weights, fc_weights)

# # Test forward pass
# # My model
# my_out = my_rnn_model(data_x)

# my_rnn_model.rnn
# my_rnn_model.rnn[0]  # first layer
# my_rnn_model.rnn[1]  # second layer
# my_rnn_model.rnn[0].W_ih.data
# my_rnn_model.rnn[1].W_ih.data

# my_rnn_model.rnn[0].W_ih.data.shape
# my_rnn_model.print_structure()
# paras = my_rnn_model.get_parameters()
# paras[0]  # parameters for the layer 0 RNN
# paras[1]  # parameters for the layer 1 RNN
# paras[2]  # parameters for the layer 2 Linear
