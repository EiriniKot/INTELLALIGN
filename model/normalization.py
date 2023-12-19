import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from typing import Optional


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
