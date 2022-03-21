import torch
from torch import nn
from itertools import chain
from typing import Union
import torch


def get_children(model: torch.nn.Module):
    # just get the children, with no regard
    # for the ordering
    children = list(model.children())
    all_ops = []
    if children == []:
        # if no children, just model
        return model
    else:
        # recursively search through children,
        # as in seq_graph()
        for c in children:
            if len(list(c.named_children())) == 0:
                all_ops.append(get_children(c))
            else:
                all_ops.extend(get_children(c))
    return all_ops


class ZeroingModelIntermediate(nn.Module):
    def __init__(self, model: torch.nn.Sequential, layer: Union[bool, int] = False):
        super(ZeroingModelIntermediate, self).__init__()

        # assume final layer if a layer is not given
        if not layer:
            self.first_half = nn.Sequential(*list(model.children())[:-1])
        # otherwise, we are attempting to zero another layer
        else:
            self.first_half = nn.Sequential(*list(model.children())[:layer])

        # freeze first half weights
        for p in self.first_half.parameters():
            p.requires_grad = False

        layer = str(layer) + "th" if layer else "final"
        self.layer = layer
        self.name = f"Zero-ing model for {self.first_half}, {layer} layer"

    def __repr__(self):
        return self.name

    def forward(self, x):
        f = self.first_half(x)
        return f


class ZeroingModelSequential(nn.Module):
    def __init__(self, model: torch.nn.Sequential, layer: Union[bool, int] = False):
        super(ZeroingModelSequential, self).__init__()

        model_ops = get_children(model)
        new_children = []
        # add layers in before linears that flattens view
        for op in model_ops:
            if type(op) == torch.nn.Linear:
                new_children.append(nn.Flatten())
            new_children.append(op)

        unfolded_model = nn.Sequential(*new_children)

        self.name = f"Zero-ing model for {model.__class__.__name__}, {layer} layer"
        # test if the same, if not use another splitting method
        x = torch.rand(1, 3, 32, 32)
        try:
            output_unfold = unfolded_model(x)
            output_normal = model(x)
        except ValueError:
            raise ValueError(f"Not able to use this class for a {self.name}")
        assert torch.isclose(
            output_normal, output_unfold
        ).all(), f"Not able to use this class for a {self.name}"

        # assume final layer if a layer is not given
        if not layer:
            self.first_half = nn.Sequential(*unfolded_model[:-1])
            self.second_half = nn.Sequential(unfolded_model[-1])
        # otherwise, we are attempting to zero another layer
        else:
            self.first_half = nn.Sequential(*unfolded_model[:layer])
            self.second_half = nn.Sequential(*unfolded_model[layer:])
        # freeze weights
        for p in chain(self.first_half.parameters(), self.second_half.parameters()):
            p.requires_grad = False

        layer = str(layer) + "th" if layer else "final"
        self.layer = layer

    def __repr__(self):
        return self.name

    def forward(self, x):
        f = self.first_half(x)
        y = self.second_half(f)
        return y, f
