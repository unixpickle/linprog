import numpy as np
import torch
import torch.nn as nn


def input_linear_program(model, inputs, out_loss_fn, epsilon=0.1, min_val=0, max_val=1):
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
        min_val: the minimum value of the adjusted inputs.
        max_val: the maximum value of the adjusted inputs.

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
            batch_x = inputs[None]
            outputs = model[:i](batch_x).detach().cpu().numpy()[0]
            shape = outputs.shape
            outputs = outputs.flatten()
            num_dims = int(np.prod(shape))
            batch_x = inputs[None].repeat(num_dims).clone().detach().requires_grad_(True)
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
    for i, x in enumerate(inputs.detach().cpu().numpy().flatten()):
        a_vec = np.zeros([input_size], dtype=np.float32)
        a_vec[i] = 1
        A_ub.append(a_vec)
        b_ub.append(min(max_val - x, epsilon))
        A_ub.append(-a_vec)
        b_ub.append(min(x - min_val, epsilon))
    in_with_grad = inputs[None].clone().detach().requires_grad_(True)
    torch.sum(out_loss_fn(model(in_with_grad))).backward()
    c = in_with_grad.grad.data.detach().cpu().numpy()
    return c, np.array(A_ub), np.array(b_ub)
