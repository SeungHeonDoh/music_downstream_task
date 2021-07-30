import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, average_precision_score

def accuracy(source, target):
    source = source.max(1)[1].long().detach().cpu().numpy()
    target = target.long().detach().cpu().numpy()
    correct = (source == target).sum().item()
    return correct/float(source.shape[0])

def get_auc(y_score, y_true):
    roc_aucs = roc_auc_score(
        y_true.flatten(0, 1).detach().cpu().numpy(),
        y_score.flatten(0, 1).detach().cpu().numpy(),
    )
    pr_aucs = average_precision_score(
        y_true.flatten(0, 1).detach().cpu().numpy(),
        y_score.flatten(0, 1).detach().cpu().numpy(),
    )
    return roc_aucs, pr_aucs

def KNN(source, target, topk=5):