from typing import Optional, Tuple
import torch.nn as nn
import torch
import copy


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
    


def LayerModifierZCA(original_op):
    
    weight = original_op.weight.data.clone()
    weight = weight.reshape(weight.shape[0], -1)
        
    U, S, Vt = torch.linalg.svd(weight, full_matrices=False)
    
    print(S)
    
    mean_sing_val = torch.mean(S)
    S[:] = mean_sing_val
        
    print(torch.var(S, unbiased=True)/torch.mean(S**2))
        
    weight_zca = U @ torch.diag_embed(S) @ Vt
        
    weight_zca = weight_zca.reshape(original_op.weight.shape) 
    
    print(torch.mean(torch.abs(original_op.weight-weight_zca)))
    
    op = copy.deepcopy(original_op)
    op.weight.data = weight_zca.clone()
    
    return op

    
#class LayerModifierZCA(nn.Module):
#    """Modifies an nn.Module from W to MW,
#    where W is the original op and M is an orthonormal matrix.
#    Projection implemented via a 1x1 convolution."""

#    def __init__(
#        self,
#         Optional[torch.device] = None
#    ):
#        super(LayerModifierZCA, self).__init__()
#        if not device:
#            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #self.original_op = original_op
#        weight = original_op.weight.data.clone()
#        weight = weight.reshape(weight.shape[0], -1)
        
#        U, S, Vt = torch.linalg.svd(weight, full_matrices=False)
        
#        mean_sing_val = torch.mean(S)
        #+S[:] = mean_sing_val
        
#        print(torch.var(S, unbiased=True)/torch.mean(S**2))
        
#        weight_zca = U @ torch.diag_embed(S) @ Vt
        
#        weight_zca = weight_zca.reshape(original_op.weight.shape) 
        
#        self.op = copy.deepcopy(original_op)
#        self.op.weight.data = weight_zca.clone()


#    def forward(self, x):
#        x = self.op(x)
#        return x
