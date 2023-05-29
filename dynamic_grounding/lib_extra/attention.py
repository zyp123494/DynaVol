"""Attention module library."""

import functools
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init


# from savi.modules import misc
# from savi.lib.utils import init_param, init_fn
import lib_extra.network as misc
from lib_extra.network import init_fn, init_param

import pdb

Shape = Tuple[int]

DType = Any
Array = torch.Tensor # np.ndarray
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet  # TODO: what is this ?
ProcessorState = ArrayTree
PRNGKey = Array
NestedDict = Dict[str, Any]


class InvertedDotProductAttention(nn.Module):
    """Inverted version of dot-product attention (softmax over query axis)."""

    def __init__(self,
                 input_size: int, # qkv_size # FIXME: added for submodules
                 output_size: int, # FIXME: added for submodules
                 num_heads: Optional[int] = 1, # FIXME: added for submodules
                 norm_type: Optional[str] = "mean", # mean, layernorm, or None
                 # multi_head: bool = False, # FIXME: can infer from num_heads.
                 epsilon: float = 1e-8,
                 dtype: DType = torch.float32,
                 weight_init = None
                 # precision # not used
                ):
        super().__init__()

        assert num_heads >= 1 and isinstance(num_heads, int)

        self.input_size = input_size
        self.output_size = output_size
        self.norm_type = norm_type
        self.num_heads = num_heads
        self.multi_head = True if num_heads > 1 else False
        self.epsilon = epsilon
        self.dtype = dtype
        self.weight_init = weight_init
        # other definitions
        self.head_dim = input_size // self.num_heads

        # submodules
        self.attn_fn = GeneralizedDotProductAttention(
            inverted_attn=True,
            renormalize_keys=True if self.norm_type == "mean" else False,
            epsilon=self.epsilon,
            dtype=self.dtype)
        if self.multi_head:
            self.dense_o = nn.Linear(input_size, output_size, bias=False)
            init_fn[weight_init['linear_w']](self.dense_o.weight)
        if self.norm_type == "layernorm":
            self.layernorm = nn.LayerNorm(output_size, eps=1e-6)

    def forward(self, query: Array, key: Array, value: Array) -> Array:
        """Computes inverted dot-product attention.

        Args:
            qk_features = [num_heads, head_dim] = qkv_dim
            query: Queries with shape of `[batch, q_num, qk_features]`.
            key: Keys with shape of `[batch, kv_num, qk_features]`.
            value: Values with shape of `[batch, kv_num, v_features]`.
            train: Indicating whether we're training or evaluating.

        Returns:
            Output of shape `[batch, n_queries, v_features]`
        """
        B, Q = query.shape[:2]

        # Apply attention mechanism
        output, attn = self.attn_fn(query=query, key=key, value=value)

        if self.multi_head:
            # Multi-head aggregation. Equivalent to concat + dense layer.
            output = self.dense_o(output.view(B, Q, self.input_size)).view(B, Q, self.output_size)
        else:
            # Remove head dimension.
            output = output.squeeze(-2)

        if self.norm_type == "layernorm":
            output = self.layernorm(output)

        return output, attn


class GeneralizedDotProductAttention(nn.Module):
    """Multi-head dot-product attention with customizable normalization axis.

    This module supports logging of attention weights in a variable collection.
    """

    def __init__(self,
                 dtype: DType = torch.float32,
                 # precision: Optional[] # not used
                 epsilon: float = 1e-8,
                 inverted_attn: bool = False,
                 renormalize_keys: bool = False,
                 attn_weights_only: bool = False
                ):
        super().__init__()

        self.dtype = dtype
        self.epsilon = epsilon
        self.inverted_attn = inverted_attn
        self.renormalize_keys = renormalize_keys
        self.attn_weights_only = attn_weights_only

    def forward(self, query: Array, key: Array, value: Array,
                train: bool = False, **kwargs) -> Array:
        """Computes multi-head dot-product attention given query, key, and value.

        Args:
            query: Queries with shape of `[batch..., q_num, num_heads, qk_features]`.
            key: Keys with shape of `[batch..., kv_num, num_heads, qk_features]`.
            value: Values with shape of `[batch..., kv_num, num_heads, v_features]`.
            train: Indicating whether we're training or evaluating.
            **kwargs: Additional keyword arguments are required when used as attention
                function in nn.MultiHeadDotPRoductAttention, but they will be ignored here.

        Returns:
            Output of shape `[batch..., q_num, num_heads, v_features]`.
        """
        del train # Unused.

        assert query.ndim == key.ndim == value.ndim, (
            "Queries, keys, and values must have the same rank.")
        assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
            "Query, key, and value batch dimensions must match.")
        assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
            "Query, key, and value num_heads dimensions must match.")
        assert key.shape[-3] == value.shape[-3], (
            "Key and value cardinality dimensions must match.")
        assert query.shape[-1] == key.shape[-1], (
            "Query and key feature dimensions must match.")

        if kwargs.get("bias") is not None:
            raise NotImplementedError(
                "Support for masked attention is not yet implemented.")

        if "dropout_rate" in kwargs:
            if kwargs["dropout_rate"] > 0.:
                raise NotImplementedError("Support for dropout is not yet implemented.")

        # Temperature normalization.
        qk_features = query.shape[-1]
        query = query / (qk_features ** 0.5) # torch.sqrt(qk_features)

        # attn.shape = (batch..., num_heads, q_num, kv_num)
        attn = torch.matmul(query.permute(0, 2, 1, 3), key.permute(0, 2, 3, 1)) # bhqd @ bhdk -> bhqk

        if self.inverted_attn:
            attention_dim = -2 # Query dim
        else:
            attention_dim = -1 # Key dim

        # Softmax normalization (by default over key dim)
        attn = torch.softmax(attn, dim=attention_dim, dtype=self.dtype)

        if self.renormalize_keys:
            # Corresponds to value aggregation via weighted mean (as opposed to sum).
            normalizer = torch.sum(attn, axis=-1, keepdim=True) + self.epsilon
            attn_n = attn / normalizer
        else:
            attn_n = attn

        if self.attn_weights_only:
            return attn_n

        # Aggregate values using a weighted sum with weights provided by `attn`
        updates = torch.einsum("bhqk,bkhd->bqhd", attn_n, value)

        return updates, attn # FIXME: return attention too, as no option for intermediate storing in module in torch.



## slot attention in uROF
class SlotAttention(nn.Module):
    def __init__(self, voxel_dim=4,
                       in_dim=32, 
                       slot_dim=64, 
                       iters=3, 
                       eps=1e-8, 
                       hidden_dim=128,
                       kernel_size=3,
                       stride=1,):
        super().__init__()
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        self.voxel_encoder = nn.Sequential(
            nn.Conv3d(voxel_dim, in_dim, kernel_size=kernel_size, stride=stride),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_dim, in_dim, kernel_size=kernel_size, stride=stride),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_dim, in_dim, kernel_size=kernel_size, stride=stride),
            nn.ReLU(inplace=True),
        )

        self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_q = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        hidden_dim = max(slot_dim, hidden_dim)

        self.to_res = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )

        self.norm_feat = nn.LayerNorm(in_dim)
        self.slot_dim = slot_dim

    def forward(self, slots, oinputs):
        """
        input:
        feat: visual feature with position information, BxNxC
        output: slots: BxKxC, attn: BxKxN
        """
        # pdb.set_trace()
        assert len(oinputs.shape) == 5
        encoder_output = self.voxel_encoder(oinputs)
        feat = encoder_output.flatten(start_dim=2).permute(0,2,1)

        B, _, _ = feat.shape

        feat = self.norm_feat(feat)
        k = self.to_k(feat)
        v = self.to_v(feat)

        attn = None
        for _ in range(self.iters):
            slot_prev = slots
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale # BxKxN
            attn = dots.softmax(dim=1) + self.eps  # BxKxN
            attn_weights = attn / attn.sum(dim=-1, keepdim=True)  # Bx(K-1)xN

            updates = torch.einsum('bjd,bij->bid', v, attn_weights)

            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slot_prev.reshape(-1, self.slot_dim)
            )
            slots = slots.reshape(B, -1, self.slot_dim)
            slots = slots + self.to_res(slots)

        return slots, attn


