from .functional import revgrad
from torch.nn import Module


class RevGrad(Module):
    def __init__(self, reverse=True, *args, **kwargs):
        """
        A gradient reversal layer.

        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """

        super().__init__(*args, **kwargs)

        self.reverse = reverse

    def forward(self, input_):
        if self.reverse:
            return revgrad(input_)
        else:
            return input_
