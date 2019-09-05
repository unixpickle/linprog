import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

MODEL_PATH = 'mnist.pt'


def create_mnist_model():
    return nn.Sequential(
        nn.Conv2d(1, 20, 5, 2),
        nn.ReLU(),
        nn.Conv2d(20, 50, 5, 2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(4*4*50, 500),
        nn.ReLU(),
        nn.Linear(500, 10),
    )


def train_mnist_model():
    model = create_mnist_model()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        return model
    train_loader = mnist_loader()
    optimizer = optim.Adam(model.parameters())
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(F.log_softmax(output, dim=-1), target)
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), MODEL_PATH)
    return model


def mnist_loader(train=True, batch=100):
    return torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=train, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=batch,
        shuffle=True)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
