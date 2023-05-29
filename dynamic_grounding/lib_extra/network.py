"""Miscellaneous modules."""

# FIXME

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# from savi.lib import utils
# from savi.lib.utils import init_fn
from lib import utils

DType = Any
Array = torch.Tensor
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
ProcessorState = ArrayTree
PRNGKey = Array
NestedDict = Dict[str, Any]

def lecun_uniform_(tensor, gain=1.):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    var = gain / float(fan_in)
    a = math.sqrt(3 * var)
    return nn.init._no_grad_uniform_(tensor, -a, a)


def lecun_normal_(tensor, gain=1., mode="fan_in"):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        scale_mode = fan_in
    elif mode == "fan_out":
        scale_mode = fan_out
    else:
        raise NotImplementedError
    var = gain / float(scale_mode)
    # constant is stddev of standard normal truncated to (-2, 2)
    std = math.sqrt(var) / .87962566103423978
    # return nn.init._no_grad_normal_(tensor, 0., std)
    kernel = torch.nn.init._no_grad_trunc_normal_(tensor, 0, 1, -2, 2) * std
    with torch.no_grad():
        tensor[:] = kernel[:]
    return tensor

def lecun_normal_fan_out_(tensor, gain=1.):
    return lecun_normal_(tensor, gain=gain, mode="fan_out")

def lecun_normal_convtranspose_(tensor, gain=1.):
    # for some reason, the convtranspose weights are [in_channels, out_channels, kernel, kernel]
    # but the _calculate_fan_in_and_fan_out treats dim 1 as fan_in and dim 0 as fan_out.
    # so, for convolution weights, have to use fan_out instead of fan_in
    # which is actually using fan_in instead of fan_out
    return lecun_normal_fan_out_(tensor, gain=gain)

init_fn = {
    'xavier_uniform': nn.init.xavier_uniform_,
    'xavier_normal': nn.init.xavier_normal_,
    'kaiming_uniform': nn.init.kaiming_uniform_,
    'kaiming_normal': nn.init.kaiming_normal_,
    'lecun_uniform': lecun_uniform_,
    'lecun_normal': lecun_normal_,
    'lecun_normal_fan_out': lecun_normal_fan_out_,
    'ones': nn.init.ones_,
    'zeros': nn.init.zeros_,
    'default': lambda x: x}

def init_param(name, gain=1.):
    assert name in init_fn.keys(), "not a valid init method"
    # return init_fn[name](tensor, gain)
    return functools.partial(init_fn[name], gain=gain)




class MLP(nn.Module):
	"""Simple MLP with one hidden layer and optional pre-/post-layernorm."""

	def __init__(self,
				 input_size: int, # FIXME: added because or else can't instantiate submodules
				 hidden_size: int,
				 output_size: int, # if not given, should be inputs.shape[-1] at forward
				 num_hidden_layers: int = 1,
				 activation_fn: nn.Module = nn.ReLU,
				 layernorm: Optional[str] = None,
				 activate_output: bool = False,
				 residual: bool = False,
				 weight_init = None
				):
		super().__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_hidden_layers = num_hidden_layers
		self.activation_fn = activation_fn
		self.layernorm = layernorm
		self.activate_output = activate_output
		self.residual = residual
		self.weight_init = weight_init

		# submodules
		## layernorm
		if self.layernorm == "pre":
			self.layernorm_module = nn.LayerNorm(input_size, eps=1e-6)
		elif self.layernorm == "post":
			self.layernorm_module = nn.LayerNorm(output_size, eps=1e-6)
		## mlp
		self.model = nn.ModuleList()
		self.model.add_module("dense_mlp_0", nn.Linear(self.input_size, self.hidden_size))
		self.model.add_module("dense_mlp_0_act", self.activation_fn())
		for i in range(1, self.num_hidden_layers):
			self.model.add_module(f"den_mlp_{i}", nn.Linear(self.hidden_size, self.hidden_size))
			self.model.add_module(f"dense_mlp_{i}_act", self.activation_fn())
		self.model.add_module(f"dense_mlp_{self.num_hidden_layers}", nn.Linear(self.hidden_size, self.output_size))
		if self.activate_output:
			self.model.add_module(f"dense_mlp_{self.num_hidden_layers}_act", self.activation_fn())
		for name, module in self.model.named_children():
			if 'act' not in name:
				# nn.init.xavier_uniform_(module.weight)
				init_fn[weight_init['linear_w']](module.weight)
				init_fn[weight_init['linear_b']](module.bias)

	def forward(self, inputs: Array, train: bool = False) -> Array:
		del train # Unused

		x = inputs
		if self.layernorm == "pre":
			x = self.layernorm_module(x)
		for layer in self.model:
			x = layer(x)
		if self.residual:
			x = x + inputs
		if self.layernorm == "post":
			x = self.layernorm_module(x)
		return x

class myGRUCell(nn.Module):
	"""GRU cell as nn.Module

	Added because nn.GRUCell doesn't match up with jax's GRUCell...
	This one is designed to match ! (almost; output returns only once)

	The mathematical definition of the cell is as follows

  	.. math::

		\begin{array}{ll}
		r = \sigma(W_{ir} x + W_{hr} h + b_{hr}) \\
		z = \sigma(W_{iz} x + W_{hz} h + b_{hz}) \\
		n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
		h' = (1 - z) * n + z * h \\
		\end{array}
	"""

	def __init__(self,
				 input_size: int,
				 hidden_size: int,
				 gate_fn = torch.sigmoid,
				 activation_fn = torch.tanh,
				 weight_init = None
				):
		super().__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.gate_fn = gate_fn
		self.activation_fn = activation_fn
		self.weight_init = weight_init

		# submodules
		self.dense_ir = nn.Linear(input_size, hidden_size)
		self.dense_iz = nn.Linear(input_size, hidden_size)
		self.dense_in = nn.Linear(input_size, hidden_size)
		self.dense_hr = nn.Linear(hidden_size, hidden_size, bias=False)
		self.dense_hz = nn.Linear(hidden_size, hidden_size, bias=False)
		self.dense_hn = nn.Linear(hidden_size, hidden_size)
		self.reset_parameters()

	def reset_parameters(self) -> None:
		recurrent_weight_init = nn.init.orthogonal_
		if self.weight_init is not None:
			weight_init = init_fn[self.weight_init['linear_w']]
			bias_init = init_fn[self.weight_init['linear_b']]
		else:
			# weight init not given
			stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
			weight_init = bias_init = lambda weight: nn.init.uniform_(weight, -stdv, stdv)
		# input weights
		weight_init(self.dense_ir.weight)
		bias_init(self.dense_ir.bias)
		weight_init(self.dense_iz.weight)
		bias_init(self.dense_iz.bias)
		weight_init(self.dense_in.weight)
		bias_init(self.dense_in.bias)
		# hidden weights
		recurrent_weight_init(self.dense_hr.weight)
		recurrent_weight_init(self.dense_hz.weight)
		recurrent_weight_init(self.dense_hn.weight)
		bias_init(self.dense_hn.bias)
	
	def forward(self, inputs, carry):
		h = carry
		# input and recurrent layeres are summed so only one needs a bias
		r = self.gate_fn(self.dense_ir(inputs) + self.dense_hr(h))
		z = self.gate_fn(self.dense_iz(inputs) + self.dense_hz(h))
		# add bias because the linear transformations aren't directly summed
		n = self.activation_fn(self.dense_in(inputs) +
							   r * self.dense_hn(h))
		new_h = (1. - z) * n + z * h
		return new_h


class GaussianStateInit(nn.Module):
    """Random state initialization with zero-mean, unit-variance Gaussian

    Note: This module does not contain any trainable parameters.
        This module also ignores any conditional input (by design).
    """

    def __init__(self,
                 num_slots,
                 slots_dimension,
                 device,
                 batch=1
                ):
        super().__init__()

        self.shape = [batch, num_slots, slots_dimension]
        self.device = device
    
    def forward(self) -> Array:
        return torch.normal(mean=torch.zeros(list(self.shape))).to(self.device)
