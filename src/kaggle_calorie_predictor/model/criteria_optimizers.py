import torch.nn as nn
import torch.optim as optim


def get_criterion(criterion_name, pad_int=None):
    criteria = {
        "CrossEntropyLoss": nn.CrossEntropyLoss(ignore_index=pad_int),
        "MSELoss": nn.MSELoss(),
    }
    return criteria[criterion_name]


def get_optimizer(optimizer_name, parameters, learning_rate):
    optimizers = {
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
    }
    return optimizers[optimizer_name](parameters, lr=learning_rate)
