"""
Full Attention Layer.
"""

from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TriangularCausalMask:
    """
    Triangular causal mask for attention mechanism.
    """

    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    """
    Full attention mechanism with optional masking and dropout.
    Args:
        mask_flag (bool): Whether to apply masking.
        factor (int): Factor for scaling the attention scores.
        scale (float): Scaling factor for attention scores.
        attention_dropout (float): Dropout rate for attention scores.
        output_attention (bool): Whether to output attention weights.
    """

    def __init__(
        self,
        mask_flag=True,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        # SDPA expects [B, H, L, E]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        is_causal = False
        if self.mask_flag and attn_mask is None:
            is_causal = True

        V = F.scaled_dot_product_attention(
            query=queries,
            key=keys,
            value=values,
            attn_mask=(
                attn_mask.mask
                if (attn_mask is not None and hasattr(attn_mask, "mask"))
                else attn_mask
            ),
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal,
            scale=scale,
        )

        V = V.transpose(1, 2)

        # NOTE keeping the self.output_attention interface for now
        if self.output_attention:
            return V.contiguous(), None
        else:
            return V.contiguous(), None


class _FullAttention(nn.Module):
    """
    Full attention mechanism with optional masking and dropout.
    Args:
        mask_flag (bool): Whether to apply masking.
        factor (int): Factor for scaling the attention scores.
        scale (float): Scaling factor for attention scores.
        attention_dropout (float): Dropout rate for attention scores.
        output_attention (bool): Whether to output attention weights.
    """

    def __init__(
        self,
        mask_flag=True,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.abs)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None
