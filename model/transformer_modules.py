import torch
from torch.nn.parameter import Parameter
from torch import nn, Tensor
from typing import Optional
from model.normalization import PaddedSequenceNormalization


class PaddedSequenceNormalization(nn.Module):
    def __init__(self, embed_dim: int, affine: bool = True, eps: float = 1e-5):
        super(PaddedSequenceNormalization, self).__init__()

        self.embed_dim = embed_dim
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.weight = Parameter(torch.empty((1, 1, embed_dim)))
            self.bias = Parameter(torch.empty((1, 1, embed_dim)))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: Tensor, padding: Optional[Tensor] = None):
        """
        Parameters
        ----------
        input: [Tensor] of shape (<batch size>, <sequence length>, <embed dim>)
        padding: [Tensor] of shape (<batch size>, <sequence length>) with 1 for each valid
            sequence item and 0 for each padded one.

        Returns
        -------
            [Tensor] Normalized sequence of same shape as input.
        """
        if padding is None:
            padding = torch.ones_like(input[:, :, 0])
        batch_size = input.shape[0]
        num_seq_items = torch.sum(padding, dim=1)

        # Compute the mean vector of each input sequence individually
        mean_input = torch.sum(input * padding.unsqueeze(-1), dim=1, keepdim=True) / num_seq_items.view(
            (batch_size, 1, 1)
        )

        output = input - mean_input

        # Compute the variance of each input sequence individually
        variance = torch.sum((output**2) * padding.unsqueeze(-1), dim=1, keepdim=True) / (num_seq_items - 1).view(
            (batch_size, 1, 1)
        )
        std = torch.sqrt(variance + self.eps)
        output = output / std
        output = output * self.weight + self.bias

        return output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads=1,
        att_dropout=0.0,
        bias=True,
        batch_first=True,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads=num_heads,
            dropout=att_dropout,
            bias=bias,
            batch_first=batch_first,
        )
        self.norm1 = PaddedSequenceNormalization(embed_dim)
        self.norm2 = PaddedSequenceNormalization(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        self.dropout = nn.Dropout(p=0.01)

    def forward(self, q, key_padding_mask):
        q_normalized = self.norm1(
            q, padding=(1.0 - key_padding_mask.float())
        )  # EQUATION 2 IN PAPER ATTENTION LEARN TO SOLVE..
        att_out, _ = self.attention(
            query=q_normalized,
            key=q_normalized,
            value=q_normalized,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        q = att_out + q
        q_mlp = self.norm2(q)  # , padding_mask)
        q_mlp = self.dropout(q_mlp)
        q_mlp = self.ff(q_mlp)

        q = q_mlp + q
        q = self.dropout(q)
        return q


#
