import torch
import numpy as np
from collections import OrderedDict

class VirtualModule:
    def __init__(self):
        self._parameter_shapes = self.get_parameter_shapes()

        self._num_parameters = 0
        for shape in self.parameter_shapes.values():
            print(shape)
            numel = np.prod(shape)
            self._num_parameters += numel

    def get_parameter_shapes(self):
        # return an OrderedDict with the parameter names and their shape
        return NotImplementedError

    # def parameter_initialization(self, num_instances):
    #     factor = 1 / ((self.num_parameters / 100) ** 0.5)
    #     initializations = []
    #     for i in range(num_instances):
    #         p = []
    #         for key, shape in self.parameter_shapes.items():
    #             p.append(torch.randn(shape).view(-1) * factor)
    #         p = torch.cat(p, dim=0)
    #         initializations.append(p)
    #     return initializations

    def split_parameters(self, p):
        if len(p.shape) == 1:
            batch_size = []
        else:
            batch_size = [p.shape[0]]
        pointer = 0
        parameters = []
        for shape in self.parameter_shapes.values():
            numel = np.prod(shape)
            x = p[..., pointer : pointer + numel].view(*(batch_size + list(shape)))
            parameters.append(x)
            pointer += numel
        return parameters

    @property
    def parameter_shapes(self):
        return self._parameter_shapes

    @property
    def num_parameters(self):
        return self._num_parameters


def linear_multi_parameter(input, weight, bias=None):
    """
    n: input batch dimension
    m: parameter batch dimension (not obligatory)
    i: input feature dimension
    o: output feature dimension
    :param input: n x (m x) i
    :param weight: (m x) o x i
    :param bias:  (m x) o
    :return: n x (m x) o
    """

    if len(weight.shape) == 2:
        # no parameter batch dimension
        x = torch.einsum("ni,oi->no", input, weight)
    elif len(input.shape) == 3:
        # parameter batch dimension for input and weights
        x = torch.einsum("nmi,moi->nmo", input, weight)
    else:
        # no parameter dimension batch for input
        x = torch.einsum("ni,moi->nmo", input, weight)
    if bias is not None:
        x = x + bias.unsqueeze(0)
    return x


class VirtualMLP(VirtualModule):
    def __init__(self, layer_sizes, nonlinearity="tanh", output_activation="linear"):
        self.layer_sizes = layer_sizes
        self.nlin_str = nonlinearity

        if nonlinearity == "tanh":
            self.nonlinearity = torch.tanh
        elif nonlinearity == "sigmoid":
            self.nonlinearity = torch.sigmoid
        else:
            self.nonlinearity = torch.relu

        if output_activation == "linear":
            self.output_activation = None
        elif output_activation == "sigmoid":
            self.output_activation = torch.sigmoid
        elif output_activation == "tanh":
            self.output_activation = torch.tanh
        elif output_activation == "softmax":
            self.output_activation = lambda x: torch.softmax(x, dim=-1)

        super(VirtualMLP, self).__init__()

    def get_parameter_shapes(self):
        parameter_shapes = OrderedDict()
        for i in range(1, len(self.layer_sizes)):
            parameter_shapes["w" + str(i)] = (
                self.layer_sizes[i],
                self.layer_sizes[i - 1],
            )
            parameter_shapes["wb" + str(i)] = (self.layer_sizes[i],)

        return parameter_shapes

    def forward(self, input, parameters, callback_func=None):
        # input_sequence: input_batch x (parameter_batch x) input_size
        # parameters: (parameter_batch x) num_params
        # return: input_batch x (parameter_batch x) output_size
        p = self.split_parameters(parameters)
        num_layers = len(self.layer_sizes) - 1
        x = input
        for l in range(0, num_layers):
            w = p[l * 2]
            a = linear_multi_parameter(x, w, bias=p[l * 2 + 1])
            if l < num_layers - 1:
                x = self.nonlinearity(a)
                if callback_func is not None:
                    callback_func(x, l)
            else:
                x = a if self.output_activation is None else self.output_activation(a)
        return x
    
    def _get_xavier_weights(self):
        wb = []
        for i in range(len(self.layer_sizes)-1):
            shape = (self.layer_sizes[i],self.layer_sizes[i+1])
            w = torch.empty(size=shape)
            torch.nn.init.xavier_uniform_(w, gain=torch.nn.init.calculate_gain(self.nlin_str))
            wb.append(w.flatten())
            wb.append(torch.zeros(shape[1]))
        return torch.hstack(wb)

class VirtualMLPPolicy(VirtualMLP):
    def __init__(self, layer_sizes, act_lim, nonlinearity, output_activation, bias=True):
        super().__init__(
            layer_sizes=layer_sizes, nonlinearity=nonlinearity, output_activation=output_activation
        )
        self.act_lim = act_lim

    def forward(self, input, parameters, callback_func=None):
        x = super().forward(input, parameters, callback_func)
        x = x * self.act_lim
        return x
