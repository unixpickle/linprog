import numpy as np
import torch.nn as nn


def input_linear_program(model, inputs, out_loss_fn, epsilon=0.1, min=0, max=1):
    """
    Create a linear program that adjusts the inputs up to
    some epsilon to maximize the output loss function,
    while not crossing ReLU boundaries.

    Args:
        model: an nn.Sequential containing linear layers
          and nn.ReLU layers, with nothing else.
        inputs: a single input to adjust. Will be batched
          in the outer dimension automatically.
        out_loss_fn: a function which takes the outputs of
          the full model and produces a batch of loss
          values to maximize.
        epsilon: the maximum delta to the inputs.
        min: the minimum value of the adjusted inputs.
        max: the maximum value of the adjusted inputs.

    Returns:
        A tuple (c, A_ub, b_ub):
          c: the linear objective function.
          A_ub: a constraint matrix.
          b_ub: the vector where A_ub*x <= b_ub.

        The resulting linear program can be solved to give
          a delta in input space that will maximize the
          loss.
    """
    input_size = int(np.prod(inputs.shape))
    A_ub = []
    b_ub = []
    for i, layer in enumerate(model):
        if isinstance(layer, nn.ReLU):
            batch_x = inputs[None].repeat(1)
            outputs = model[:i](batch_x).detach().cpu().numpy()[0]
            shape = outputs.shape
            outputs = outputs.flatten()
            num_dims = int(np.prod(shape))
            batch_x = x[None].repeat(num_dims).clone().detach().requires_grad_(True)
            batch_y = model[:i](batch_x)
            batch_y.view(num_dims, num_dims).backward(torch.eye(num_dims))
            gradient = batch_x.grad.data.view(num_dims, input_size).detach().cpu().numpy()
            for i in range(num_dims):
                if outputs[i] < 0:
                    A_ub.append(gradient[i])
                    b_ub.append(-outputs[i])
                else:
                    A_ub.append(gradient[i] * -1)
                    b_ub.append(outputs[i])
    c = None  # TODO: gradient here.
    return c, A_ub, b_ub
