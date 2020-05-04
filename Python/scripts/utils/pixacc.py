import numpy as np
import torch


def pixacc(pred, label):
    pred = pred.view(-1)
    label = label.view(-1)

    corrects = torch.sum(pred == label)
    acc = int(corrects)/len(pred)


    return acc