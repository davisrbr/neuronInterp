from typing import Optional, Tuple
import torch.nn as nn
import torch


class LayerModifier(nn.Module):
    """Modifies an nn.Module from W to MW,
    where W is the original op and M is an orthonormal matrix.
    Projection implemented via a 1x1 convolution."""

    def __init__(
        self,
        original_op: nn.Module,
        dims: Tuple,
        rand_ortho: Optional[torch.tensor] = None,
        device: Optional[torch.device] = None
    ):
        super(LayerModifier, self).__init__()
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not rand_ortho:
            rand_ortho_weight = torch.nn.init.orthogonal_(torch.empty((dims, dims))).to(
                device
            )
            rand_ortho_weight = torch.nn.Parameter(rand_ortho_weight[..., None, None])
            rand_ortho_weight.requires_grad = False

            conv1x1_ortho = nn.Conv2d(
                in_channels=dims, out_channels=dims, kernel_size=1, bias=False
            )
            conv1x1_ortho.weight = rand_ortho_weight

        if isinstance(original_op, torch.nn.Conv2d):
            self.rand_ortho = lambda x: conv1x1_ortho(x)
        elif isinstance(
            original_op, torch.nn.Linear
        ):  # unsqueeze final two dims if linear
            self.rand_ortho = lambda x: conv1x1_ortho(x[..., None, None]).reshape(
                -1, dims
            )
        else:
            raise NotImplementedError(
                f"Only for linear and conv2d, not {type(original_op)}"
            )

        self.original_op = original_op

    def forward(self, x):
        x = self.original_op(x)
        x = self.rand_ortho(x)
        return x
