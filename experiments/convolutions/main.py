from scipy.optimize import linprog
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
        grad, constraint_coeffs, constraint_bounds = input_linear_program(model, sample, loss_fn)
        print('running linprog with %d constraints' % constraint_coeffs.shape[0])
        solution = linprog(grad, A_ub=constraint_coeffs, b_ub=constraint_bounds,
                           bounds=(None, None))
        print(solution.x)


if __name__ == '__main__':
    main()
