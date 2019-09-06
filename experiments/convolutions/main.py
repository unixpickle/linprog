from scipy.optimize import linprog
import numpy as np
import torch
import torch.nn.functional as F

from classifier import train_mnist_model, mnist_loader
from program import input_linear_program


def main():
    model = train_mnist_model()
    dataset = mnist_loader(train=False, batch=1).__iter__()
    for i in range(10):
        sample, label = next(dataset)
        sample = sample[0]
        label = label.item()

        def loss_fn(x):
            return -F.log_softmax(x, dim=0)[label]
        print('creating linear program...')
        grad, constraint_coeffs, constraint_bounds = input_linear_program(model, sample, loss_fn)
        print('running linprog with %d constraints...' % constraint_coeffs.shape[0])
        solution = linprog(-grad, A_ub=constraint_coeffs, b_ub=constraint_bounds,
                           bounds=(None, None))
        new_input = sample + torch.from_numpy(solution.x).float().view(*sample.shape)
        fgsm_input = fgsm(sample, grad)
        old_prob = F.softmax(model(sample[None]), dim=-1)[0, label]
        new_prob = F.softmax(model(new_input[None]), dim=-1)[0, label]
        fgsm_prob = F.softmax(model(fgsm_input[None]), dim=-1)[0, label]
        print('correct prob went from %f to %f (fgsm %f)' % (old_prob, new_prob, fgsm_prob))


def fgsm(sample, grad, epsilon=0.1, min_val=0, max_val=1):
    delta = np.sign(grad) * epsilon
    return torch.clamp(sample + torch.from_numpy(delta).float().view(*sample.shape),
                       min_val, max_val)


if __name__ == '__main__':
    main()
