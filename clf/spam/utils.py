import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
from torch.utils.data.sampler import WeightedRandomSampler


def imshow(inp, title=None):
    inp = torchvision.utils.make_grid(inp)
    plt.figure(figsize=(14,14))
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def get_weighted_sampler(label, ratio):
    weight = np.bincount(label)
    weight = 1 / weight
    samples_weight = np.array([weight[t] for t in label])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), int(len(samples_weight)*ratio))
    return sampler