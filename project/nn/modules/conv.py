import numpy as np
from nn.modules.parameters import Parameter


class Conv1D:

    """
    no padding, no dilation

    General expressions:
    output= Z = X * W + b,  * is cross-correlation
    output_size = floor(1 + (n_rows_in + pad_left + pad_right - kernel_size) / stride)

    Args:
        in_channel  (int): number of input channels
        out_channel (int): number of output channels
        kernel_size (int): the size of the kernel
        stride      (int): stride
        weight_init_fn (fn): weight initializer. Defaults to None.
        bias_init_fn   (fn): bias initializer. Defaults to None.

    Attrs:
        in_channel: in_channel
        out_channel: out_channel
        kernel_size: kernel_size
        stride: stride
        W (obj-Parameter <nn.modules.parameters>): weight matrix (out_channel, in_channel, kernel_size)
        b (obj-Parameter <nn.modules.parameters>): bias (out_channel, 1)
        parameters: a dict {"W": W, "b": b}
        x (np.array): input data, (batch_size, in_channel, input_size)

    Methods:
        forward
        backward

    """

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride,
        weight_init_fn=None,
        bias_init_fn=None,
    ):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        self.W = Parameter(np.zeros((out_channel, in_channel, kernel_size)))
        self.b = Parameter(np.zeros((out_channel, 1)))

        if weight_init_fn is not None:
            self.W.data = weight_init_fn(self.W.data)
        if bias_init_fn is not None:
            self.b.data = bias_init_fn(self.b.data)

        self.parameters = {"W": self.W, "b": self.b}
        self.x = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
                            X * W + b,  * is cross-correlation

        """

        self.x = x

        batch_size, in_channel, input_size = self.x.shape
        output_size = (input_size - self.kernel_size) // self.stride + 1
        out = np.zeros([batch_size, self.out_channel, output_size])

        for b in range(batch_size):
            for f in range(self.out_channel):
                for i in range(output_size):
                    out[b, f, i] = np.tensordot(
                        self.W.data[f, :, :],
                        self.x[
                            b,
                            :,
                            i * self.stride : i * self.stride + self.kernel_size,
                        ],
                        axes=[(0, 1), (0, 1)],
                    )
                    # alternatively,
                    # out[b, f, i]=np.sum(self.W[f,:,:] * x[b,:,i * self.stride:i * self.stride+self.kernel_size])
                out[b, f] += self.b.data[f, 0]
        return out

    def backward(self, dLdZ):
        """
        This function outputs dx, and update self.W.grad and self.b.grad as byproduct.
        i.e., dLdZ --> (dLdW, dLdb)-->dx
        (dLdW, dLdb) are populated as attributes

        Note:
            we accumulate gradients on W and b!
            We do not accumualte gradient on x

        Argument:
            dLdZ (np.array): dLdZ, Z is affine values, (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)

        """
        # Calculate dW
        batch_size, out_channel, output_size = dLdZ.shape
        for batch in range(batch_size):
            for out_c in range(self.out_channel):
                for in_c in range(self.in_channel):
                    for i in range(self.kernel_size):
                        for out in range(output_size):
                            self.W.grad[out_c, in_c, i] += (
                                self.x[batch, in_c, i + self.stride * out]
                                * dLdZ[batch, out_c, out]
                            )
                            # note: we did not divide it by batch_size

        # Calculate db
        self.b.grad += np.sum(dLdZ, axis=(0, 2)).reshape(out_channel, 1)
        # note: we did not divide it by batch_size

        # Calculate dx
        # zero out dx
        dx = np.zeros(self.x.shape)

        # calculating the input shape as follow is wrong, as this fails to
        # account for the integer function when computing output shape)
        # input_size = int((output_size - 1) * self.stride + kernel_size)

        for batch in range(batch_size):
            for in_c in range(self.in_channel):
                for out_c in range(self.out_channel):
                    for s in range(output_size):
                        for k in range(self.kernel_size):
                            dx[batch, in_c, self.stride * s + k] += (
                                dLdZ[batch, out_c, s] * self.W.data[out_c, in_c, k]
                            )

        return dx


class Conv1D_dilation:

    """
    with padding on X, and dilation on W

    General expressions:
    output= Z = X * W + b,  * is cross-correlation

    output size = [(input size padded - kernel dilated)//stride] + 1
    with input size padded = input size + 2 * padding
    kernel dilated = (kernel size - 1) * (dilation - 1) + kernel size
    equivalently,
    output size = [(input size + 2 * padding - dilation * (kernel size - 1) - 1)//stride] + 1

    Args:
        in_channel  (int): number of input channels
        out_channel (int): number of output channels
        kernel_size (int): the size of the kernel
        stride      (int): stride
        padding     (int):  =0
        dilation    (int): =1
        weight_init_fn (fn): weight initializer. Defaults to None.
        bias_init_fn   (fn): bias initializer. Defaults to None.

    Attrs:
        in_channel: in_channel
        out_channel: out_channel
        kernel_size: kernel_size
        stride: stride
        W (obj-Parameter <nn.modules.parameters>): weight matrix (out_channel, in_channel, kernel_size)
        b (obj-Parameter <nn.modules.parameters>): bias (out_channel, 1)
                note: Pytorch nn.module, bias is of shape (out_features, )
        parameters: a dict {"W": W, "b": b}
        W_dilated (np.array): dilated W
        dW_dilated (np.array):
        x (np.array): input data, (batch_size, in_channel, input_size)

    Methods:
        forward
        backward

    """

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride,
        padding=0,
        dilation=1,
        weight_init_fn=None,
        bias_init_fn=None,
    ):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # dilation
        self.kernel_dilated = (self.kernel_size - 1) * self.dilation + 1

        self.W = Parameter(np.zeros((out_channel, in_channel, kernel_size)))
        self.b = Parameter(np.zeros((out_channel, 1)))

        if weight_init_fn is not None:
            self.W.data = weight_init_fn(self.W.data)
        if bias_init_fn is not None:
            self.b.data = bias_init_fn(self.b.data)

        self.parameters = {"W": self.W, "b": self.b}

        self.W_dilated = np.zeros(
            (self.out_channel, self.in_channel, self.kernel_dilated)
        )

        self.dW_dilated = np.zeros_like(self.W_dilated)

        self.x = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x   (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
                            X * W + b,  * is cross-correlation

        """

        self.x = x

        # add padding to x
        x_padded = np.pad(
            x, [(0, 0), (0, 0), (self.padding, self.padding)], mode="constant"
        )

        # dilation on W -> first upsample the W -> computation: k_new = (k-1) * (dilation-1) + k
        for cout in range(self.out_channel):
            for cin in range(self.in_channel):
                for i in range(self.kernel_size):
                    self.W_dilated[cout, cin, self.dilation * i] = self.W.data[
                        cout, cin, i
                    ]

        batch_size, in_channel, input_size = self.x.shape

        # Calculate output size using formula
        # [(n + 2p - d(k-1) - 1)/s] + 1
        output_size = (
            int(
                (
                    input_size
                    + 2 * self.padding
                    - self.dilation * (self.kernel_size - 1)
                    - 1
                )
                / self.stride
            )
            + 1
        )

        out = np.zeros([batch_size, self.out_channel, output_size])

        # recall: x_padded shape (batch_size, in_channel, input_size)
        # W_dialted shape  (out_channel, in_channel, kernel_size_d)
        # out shape (batch_size, out_channel, output_size)
        for b in range(batch_size):
            for f in range(self.out_channel):
                for i in range(output_size):
                    out[b, f, i] = np.tensordot(
                        self.W_dilated[f, :, :],
                        x_padded[
                            b,
                            :,
                            i * self.stride : i * self.stride + self.kernel_dilated,
                        ],
                        axes=[(0, 1), (0, 1)],
                    )
                    # alternatively,
                    # out[b, f, i]=np.sum(self.W_dilated[f,:,:] * x[b,:,i * self.stride:i * self.stride+ self.kernel_dilated])
                out[b, f] += self.b.data[f, 0]

        return out

    def backward(self, dLdZ):
        """
        This function outputs dx, and update self.dW and self.db as byproduct.

        Note: We accumulate gradients (i.e. W.grad, and b.grad).
        We do not accumulate gradient for x, W_dialted

        Argument:
            dLdZ (np.array): dLdZ, Z is affine values, (batch_size, out_channel, output_size)
        Return:
            dx   (np.array): (batch_size, in_channel, input_size)

        """

        batch_size, out_channel, output_size = dLdZ.shape

        # pad input tensor
        x_padded = np.pad(
            self.x,
            ((0, 0), (0, 0), (self.padding, self.padding)),
            mode="constant",
        )

        # dilate kernel
        W_dilated = np.zeros((self.out_channel, self.in_channel, self.kernel_dilated))

        # Perform the dilation on the kernel W
        for c_out in range(self.out_channel):
            for c_in in range(self.in_channel):
                for i in range(self.kernel_size):
                    W_dilated[c_out, c_in, i * self.dilation] = self.W.data[
                        c_out, c_in, i
                    ]

        # note: we do not accumulate gradient on x， W_dilated
        # here zeroing it out
        # our optimizers.zero_grad() will zero out W.grad and b.grad
        self.dW_dilated = np.zeros_like(W_dilated)
        dx_padded = np.zeros(x_padded.shape)

        # Compute dW, db and dx_padded
        self.b.grad += np.sum(dLdZ, axis=(0, 2)).reshape(out_channel, 1)

        for b in range(batch_size):
            for k in range(out_channel):
                for i in range(output_size):
                    # calculate gradients for dilated kernel W_dilated
                    self.dW_dilated[k] += (
                        dLdZ[b, k, i]
                        * x_padded[
                            b,
                            :,
                            i * self.stride : i * self.stride + self.kernel_dilated,
                        ]
                    )

                    # calculate gradient of padded input tensor x_padded
                    dx_padded[
                        b, :, i * self.stride : i * self.stride + self.kernel_dilated
                    ] += (dLdZ[b, k, i] * W_dilated[k])

        # Crop the resultant gradients to match the original shapes
        # zero padding had added additional rows and columns to them.
        if self.padding > 0:
            dx = dx_padded[:, :, self.padding : -self.padding]
        else:
            dx = dx_padded

        self.W.grad += self.dW_dilated[:, :, : self.kernel_dilated : self.dilation]

        return dx


class Conv2D:
    """
    no padding, no dilation.

    output =Z = X * W + b,  * is cross-correlation
    X shape NCHW (batch_size, in_channel, input_height, input_width)
    output_height = floor(1 + (input_height  - kernel_height) / stride)
    output_width = floor(1 + (input_width  - kernel_width) / stride)

    We only consider square kernel here (i.e., H=W)

    Args:
        in_channel  (int): number of input channels
        out_channel (int): number of output channels
        kernel_size (int): the size of the kernel (height=width=size)
        stride      (int): stride
        weight_init_fn (fn): weight initializer. Defaults to None.
        bias_init_fn   (fn): bias initializer. Defaults to None.

    Attrs:
        in_channel: in_channel
        out_channel: out_channel
        kernel_size: kernel_size
        stride: stride
        W (obj-Parameter <nn.modules.parameters>): weight matrix (out_channel, in_channel, kernel_size, kernel_size)
        b (obj-Parameter <nn.modules.parameters>): bias (out_channel, 1)
        parameters: a dict {"W": W, "b": b}
        x (np.array): input data, (batch_size, in_channel, input_height, input_width)

    Methods:
        forward
        backward
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride,
        weight_init_fn=None,
        bias_init_fn=None,
    ):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        self.W = Parameter(
            np.zeros((out_channel, in_channel, kernel_size, kernel_size))
        )
        self.b = Parameter(np.zeros((out_channel, 1)))

        if weight_init_fn is not None:
            self.W.data = weight_init_fn(self.W.data)
        if bias_init_fn is not None:
            self.b.data = bias_init_fn(self.b.data)

        self.parameters = {"W": self.W, "b": self.b}
        self.x = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x   (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            out (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.x = x

        batch_size, in_channels, input_height, input_width = x.shape
        out_channels, _, kernel_height, kernel_width = self.W.shape
        output_height = int((input_height - kernel_height) / self.stride) + 1
        output_width = int((input_width - kernel_width) / self.stride) + 1

        out = np.zeros((batch_size, out_channels, output_height, output_width))

        for b in range(batch_size):
            for k in range(out_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        window = x[
                            b,
                            :,
                            i * self.stride : i * self.stride + kernel_height,
                            j * self.stride : j * self.stride + kernel_width,
                        ]
                        # window shape (in_channels, kernel_height, kernel_width)
                        # self.W[k] shape (in_channels, kernel_height, kernel_width)
                        out[b, k, i, j] = (
                            np.sum(window * self.W.data[k]) + self.b.data[k, 0]
                        )

        return out

    def backward(self, dLdZ):
        """
        This function outputs dx, and update self.dW and self.db as byproduct.

        Note: we the accumulate gradients on W and b!
        We do not accumualte gradient on x

        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dx   (np.array): (batch_size, in_channels, input_height, input_width)

        """
        batch_size, out_channels, output_height, output_width = dLdZ.shape
        _, in_channels, kernel_height, kernel_width = self.W.shape

        # zero out dx
        dx = np.zeros(self.x.shape)

        # calculating the input shape as follow is wrong! as this fails to
        # account for the integer function when computing output shape)
        # input_height = int((output_height - 1) * self.stride + kernel_height)
        # input_width = int((output_width - 1) * self.stride + kernel_width)

        for b in range(batch_size):
            for k in range(out_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        # compute dx
                        dx[
                            b,
                            :,
                            i * self.stride : i * self.stride + kernel_height,
                            j * self.stride : j * self.stride + kernel_width,
                        ] += (
                            dLdZ[b, k, i, j]
                            * self.W.data[k]
                            # self.W[k] shape (in_channels, kernel_height, kernel_width)
                        )
                        # LHS of "+="" has shape (in_channels, kernel_height, kernel_width)

                        # compute dW and db
                        window = self.x[
                            b,
                            :,
                            i * self.stride : i * self.stride + kernel_height,
                            j * self.stride : j * self.stride + kernel_width,
                        ]
                        # k-th out_channel
                        self.W.grad[k] += dLdZ[b, k, i, j] * window
                        self.b.grad[k, 0] += dLdZ[b, k, i, j]

        return dx


class Conv2D_dilation:
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride,
        padding=0,
        dilation=1,
        weight_init_fn=None,
        bias_init_fn=None,
    ):
        """
        A general class allowing padding of x, and dilation of W.

        Much like Conv2D, but take two attributes into consideration: padding and dilation.
        HINT: the only difference are the padded input and dilated kernel.

        output = Z = X * W + b,  * is cross-correlation
            X shape NCHW (batch_size, in_channel, input_height, input_width)
            output_height = floor(1 + (input_height  - kernel_height) / stride)
            output_width = floor(1 + (input_width  - kernel_width) / stride)

        Default is =1, i.e., no dilation.

        output size = [(input size padded - kernel dilated)//stride] + 1
        with input size padded = input size + 2 * padding
             kernel dilated = (kernel size - 1) * (dilation - 1) + kernel size
        equivalently,
        output size = [(input size + 2 * padding - dilation * (kernel size - 1) - 1)//stride] + 1

        Args:
            in_channel  (int): number of input channels
            out_channel (int): number of output channels
            kernel_size (int): the size of the kernel (height=width=size)
            stride   (int): stride factor (1: default stride)
            padding  (int): padding (0: no padding)
            dilation (int): dilation factor (1: no dilation)
            weight_init_fn (fn): weight initializer. Defaults to None.
            bias_init_fn   (fn): bias initializer. Defaults to None.

        Attrs:
            in_channel: in_channel
            out_channel: out_channel
            kernel_size: kernel_size
            stride: stride
            padding (int): padding (0: no padding)
            dilation (int): dilation factor (1: no dilation)
            kernel_dilated (int): the size of the dilated kernel
            W (obj-Parameter <nn.modules.parameters>): weight matrix (out_channel, in_channel, kernel_size, kernel_size)
            b (obj-Parameter <nn.modules.parameters>): bias (out_channel, 1)
            parameters: a dict {"W": W, "b": b}
            W_dilated (np.array): dilated W
            dW_dilated (np.array):
            x (np.array): input data, (batch_size, in_channel, input_height, input_width)

        Methods:
            forward
            backward
        """

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # dilation
        self.kernel_dilated = (self.kernel_size - 1) * self.dilation + 1

        self.W = Parameter(
            np.zeros((out_channel, in_channel, kernel_size, kernel_size))
        )
        self.b = Parameter(np.zeros((out_channel, 1)))

        if weight_init_fn is not None:
            self.W.data = weight_init_fn(self.W.data)
        if bias_init_fn is not None:
            self.b.data = bias_init_fn(self.b.data)

        self.W_dilated = np.zeros(
            (
                self.out_channel,
                self.in_channel,
                self.kernel_dilated,
                self.kernel_dilated,
            )
        )

        self.dW_dilated = np.zeros_like(self.W_dilated)
        self.x = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x   (np.array): (batch_size, in_channel, input_height, input_width)
        Return:
            out (np.array): (batch_size, out_channel, output_height, output_width)

        """
        self.x = x

        # padding x with self.padding parameter (use np.pad())
        x_padded = np.pad(
            self.x,
            (
                (0, 0),
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
            ),
            mode="constant",
        )


        # pad_width: {sequence, array_like, int}
        # Number of values padded to the edges along each axis.
        # e.g.: ((before_1, after_1), … (before_N, after_N)) unique pad widths for each axis.
        # mode="constant", the padded value is a constant
        # the default argument "constant_values" is 0.

        # dilation on W -> first upsample the W -> computation: k_new = (k-1) * (dilation-1) + k = (k-1) * d + 1
        for cout in range(self.out_channel):
            for cin in range(self.in_channel):
                for i in range(self.kernel_size):
                    for j in range(self.kernel_size):
                        self.W_dilated[
                            cout, cin, self.dilation * i, self.dilation * j
                        ] = self.W.data[cout, cin, i, j]

        # regular forward, just like Conv2d().forward()
        batch_size, in_channels, in_h, in_w = self.x.shape
        out_channels, _, kernel_h_d, kernel_w_d = self.W_dilated.shape

        # obtain output height and width
        out_h = (
            int(
                (in_h + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
                / self.stride
            )
            + 1
        )
        out_w = (
            int(
                (in_w + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
                / self.stride
            )
            + 1
        )

        out = np.zeros((batch_size, out_channels, out_h, out_w))

        for b in range(batch_size):
            for k in range(out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        window = x_padded[
                            b,
                            :,
                            i * self.stride : i * self.stride + kernel_h_d,
                            j * self.stride : j * self.stride + kernel_w_d,
                        ]
                        # window shape (in_channels, kernel_h_d, kernel_w_d)
                        # self.W_dilated[k] shape (in_channels, kernel_h_d, kernel_w_d)
                        out[b, k, i, j] = (
                            np.sum(window * self.W_dilated[k]) + self.b.data[k, 0]
                        )

        return out

    def backward(self, dLdZ):
        """
        This function outputs dx, and update self.dW and self.db as byproduct.

        Note: the accumulated gradients (i.e. W.grad, and b.grad).
        However, we do not accumulate gradient for x, W_dialted

        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dx   (np.array): (batch_size, in_channels, input_height, input_width)

        """

        batch_size, out_channels, output_height, output_width = dLdZ.shape
        _, in_channels, kernel_height, kernel_width = self.W.shape

        # pad input tensor
        x_padded = np.pad(
            self.x,
            (
                (0, 0),
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
            ),
            mode="constant",
        )

        # dilate kernel
        W_dilated = np.zeros(
            (
                self.out_channel,
                self.in_channel,
                self.kernel_dilated,
                self.kernel_dilated,
            )
        )

        # Perform the 2D dilation on the kernel W
        for c_out in range(self.out_channel):
            for c_in in range(self.in_channel):
                for i in range(self.kernel_size):
                    for j in range(self.kernel_size):
                        W_dilated[
                            c_out, c_in, i * self.dilation, j * self.dilation
                        ] = self.W.data[c_out, c_in, i, j]

        # note: we do not accumulate gradient on x, W_dilated
        # here zeroing it out
        # our optimizers.zero_grad() will zero out W.grad and b.grad
        self.dW_dilated = np.zeros_like(W_dilated)
        dx_padded = np.zeros(x_padded.shape)

        # Compute dW, db and dx_padded
        self.b.grad += np.sum(dLdZ, axis=(0, 2, 3)).reshape(out_channels, 1)

        for b in range(batch_size):
            for k in range(out_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        # calculate gradients for dilated kernel W_dilated
                        self.dW_dilated[k] += (
                            dLdZ[b, k, i, j]
                            * x_padded[
                                b,
                                :,
                                i * self.stride : i * self.stride + self.kernel_dilated,
                                j * self.stride : j * self.stride + self.kernel_dilated,
                            ]
                        )

                        # calculate gradient of padded input tensor x_padded
                        dx_padded[
                            b,
                            :,
                            i * self.stride : i * self.stride + self.kernel_dilated,
                            j * self.stride : j * self.stride + self.kernel_dilated,
                        ] += (
                            dLdZ[b, k, i, j] * W_dilated[k]
                        )

        # Crop the resultant gradients to match the original shapes
        # zero padding had added additional rows and columns to them.
        if self.padding > 0:
            dx = dx_padded[
                :, :, self.padding : -self.padding, self.padding : -self.padding
            ]
        else:
            dx = dx_padded

        self.W.grad += self.dW_dilated[
            :,
            :,
            : self.kernel_dilated : self.dilation,
            : self.kernel_dilated : self.dilation,
        ]

        return dx


class Flatten1D:
    """
    1D Flatten layer that takes input x and flatten it
        x   (np.array): (batch_size, in_channel, in_width)
        out (np.array): (batch_size, in_channel * in_width)
    """

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x   (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in_width)
        """
        self.b, self.c, self.w = x.shape
        return x.reshape(self.b, self.c * self.w)

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channel * in width)
        Return:
            dx  (np.array): (batch size, in channel, in width)
        """
        # Return the derivative of the loss with respect to the flatten
        # layer input
        # Calculate dX
        dx = np.reshape(dLdZ, (self.b, self.c, self.w))
        return dx


class Flatten2D:
    """
    2D Flatten layer that takes input x and flatten it
    x   (np.array): (batch_size, in_channel, in_height, in_width)
    out (np.array): (batch_size, in_channel * in_height * in_width)
    """

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x   (np.array): (batch_size, in_channel, in_height, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in_height * in_width)
        """
        self.b, self.c, self.h, self.w = x.shape
        return x.reshape(self.b, self.c * self.h * self.w)

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channel * in_height * in_width)
        Return:
            dx   (np.array): (batch_size, in_channel, in_height, in_width)
        """
        # Return the derivative of the loss with respect to the flatten
        # layer input
        # Calculate dx
        dx = np.reshape(dLdZ, (self.b, self.c, self.h, self.w))
        return dx


class Pool1D:
    """
    1D pooling layer that takes input x and performs either max pooling or average pooling
        x          (np.array): (batch_size, in_channel, in_width)
        out        (np.array): (batch_size, out_channel, out_width)

    we set in_channel= out_channel

    Args:
        pool_size  (int):      size of the pooling window
        stride     (int):      stride of the pooling operation. Default = pool_size
        mode       (str):      'max' or 'average' pooling
    """

    def __init__(self, pool_size=2, stride=None, mode="max"):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.mode = mode

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x   (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, out_channel, out_width)
            idx (np.array): (batch_size, out_channel, out_width)
        """
        self.x = x
        self.b, self.c, self.w = x.shape
        out_w = (self.w - self.pool_size) // self.stride + 1
        out = np.zeros((self.b, self.c, out_w))
        idx = np.zeros_like(out, dtype=int)

        for i in range(out_w):
            # i index in the output
            start_idx = i * self.stride
            end_idx = start_idx + self.pool_size

            if self.mode == "max":
                # Find the index corresponding to the maximum value
                # then obtain the value using the index
                idx[:, :, i] = np.argmax(self.x[:, :, start_idx:end_idx], axis=2)
                # rhs has shape is (self.b, self.c)
                out[:, :, i] = np.max(self.x[:, :, start_idx:end_idx], axis=2)
                # same as
                # out[:, :, i] = x[:, :, start_idx:end_idx][np.arange(x.shape[0])[:, None], np.arange(x.shape[1]), idx[:, :, i]]

            elif self.mode == "average":
                out[:, :, i] = np.mean(self.x[:, :, start_idx:end_idx], axis=2)

        return out

    def backward(self, dLdA):
        """
        Argument:
            dLdA (np.array): (batch_size, out_channel, out_width)
        Return:
            dx   (np.array): (batch_size, in_channel, in_width)

        """

        out_w = (self.w - self.pool_size) // self.stride + 1
        dx = np.zeros_like(self.x)

        for i in range(out_w):
            start_idx = i * self.stride
            end_idx = start_idx + self.pool_size

            if self.mode == "max":
                # loop over self.b, self.c
                for b in range(self.b):
                    for c in range(self.c):
                        max_idx = np.argmax(self.x[b, c, start_idx:end_idx])
                        dx[b, c, start_idx + max_idx] += dLdA[b, c, i]

            elif self.mode == "average":
                # Compute gradient of loss with respect to all elements in each pooling window
                grad = dLdA[:, :, i][:, :, None] / (self.pool_size)
                # grad shape: (batch_size, out_channel, 1)

                # Add gradient to all elements in each pooling window (through broadcasting)
                dx[:, :, start_idx:end_idx] += grad
                # dx[:, :, start_idx:end_idx] shape (batch_size, out_channel, pool_size)

        return dx


class Pool2D:
    """
    2D pooling layer that takes input x and performs either max pooling or average pooling
        x          (np.array): (batch_size, in_channel, height, width)
        out        (np.array): (batch_size, out_channel, out_height, out_width)

    we set in_channel= out_channel

    Args:
        pool_size  (int or list): size of the pooling window. If int, same value is used for height and width.
        stride     (int or list): stride of the pooling operation. Default = pool_size. If int, same value is used for height and width.
        mode       (str):         'max' or 'average' pooling
    """

    def __init__(self, pool_size=2, stride=None, mode="max"):
        self.pool_size = (
            pool_size if isinstance(pool_size, list) else (pool_size, pool_size)
        )
        self.stride = stride if stride is not None else self.pool_size
        self.mode = mode

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x   (np.array): (batch_size, in_channel, height, width)
        Return:
            out (np.array): (batch_size, out_channel, out_height, out_width)
            idx (np.array): (batch_size, out_channel, out_height, out_width)
        """
        self.x = x
        self.b, self.c, self.h, self.w = x.shape
        out_h = (self.h - self.pool_size[0]) // self.stride[0] + 1
        out_w = (self.w - self.pool_size[1]) // self.stride[1] + 1
        out = np.zeros((self.b, self.c, out_h, out_w))
        idx = np.zeros_like(out, dtype=int)

        for i in range(out_h):
            for j in range(out_w):
                # i, j index in the output
                start_idx_h = i * self.stride[0]
                end_idx_h = start_idx_h + self.pool_size[0]
                start_idx_w = j * self.stride[1]
                end_idx_w = start_idx_w + self.pool_size[1]

                if self.mode == "max":
                    # obtain the maximum value in the window
                    out[:, :, i, j] = np.max(
                        self.x[:, :, start_idx_h:end_idx_h, start_idx_w:end_idx_w],
                        axis=(2, 3),
                    )

                elif self.mode == "average":
                    out[:, :, i, j] = np.mean(
                        self.x[:, :, start_idx_h:end_idx_h, start_idx_w:end_idx_w],
                        axis=(2, 3),
                    )

        return out

    def backward(self, dLdA):
        """
        Argument:
            dLdA (np.array): (batch_size, out_channel, out_height, out_width)
        Return:
            dx   (np.array): (batch_size, in_channel, height, width)
        """
        out_h = (self.h - self.pool_size[0]) // self.stride[0] + 1
        out_w = (self.w - self.pool_size[1]) // self.stride[1] + 1
        dx = np.zeros_like(self.x)

        for i in range(out_h):
            for j in range(out_w):
                start_idx_h = i * self.stride[0]
                end_idx_h = start_idx_h + self.pool_size[0]
                start_idx_w = j * self.stride[1]
                end_idx_w = start_idx_w + self.pool_size[1]

                if self.mode == "max":
                    # loop over self.b, self.c
                    for b in range(self.b):
                        for c in range(self.c):
                            max_idx_h, max_idx_w = np.unravel_index(
                                np.argmax(
                                    self.x[
                                        b,
                                        c,
                                        start_idx_h:end_idx_h,
                                        start_idx_w:end_idx_w,
                                    ]
                                ),
                                self.pool_size,
                            )
                            # np.argmax with axis None retuns index into the flattened array
                            # np.unravel_index converts a flat index or array of
                            # flat indices into a tuple of coordinate arrays.
                            dx[
                                b, c, start_idx_h + max_idx_h, start_idx_w + max_idx_w
                            ] += dLdA[b, c, i, j]

                elif self.mode == "average":
                    grad = dLdA[:, :, i, j][:, :, None, None] / (
                        self.pool_size[0] * self.pool_size[1]
                    )
                    # grad.shape (batch_size, out_channel, 1, 1)
                    dx[:, :, start_idx_h:end_idx_h, start_idx_w:end_idx_w] += grad

        return dx
