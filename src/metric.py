import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, average_precision_score

def accuracy(source, target):
    source = source.max(1)[1].long().detach().cpu().numpy()
    target = target.long().detach().cpu().numpy()
    correct = (source == target).sum().item()
    return correct/float(source.shape[0])

def get_aucs(gt_array, est_array):
    roc_aucs = roc_auc_score(gt_array, est_array, average="macro")
    pr_aucs = average_precision_score(gt_array, est_array, average="macro")
    return roc_aucs, pr_aucs

def multilabel_recall(sim_matrix, binary_labels, top_k):
    results = []
    for idx in tqdm(range(len(sim_matrix))):
        sorted_idx = np.argsort(sim_matrix[idx])[::-1][1:top_k+1]
        
        y_q = binary_labels[idx]
        
        sum_list = []
        for jdx in sorted_idx:
            y_i = binary_labels[jdx]
            sum_list.append(y_i)
        
        predict_sum = np.array(sum_list).sum(axis=0)
        predict_sum_binary = np.where(predict_sum < 1, predict_sum, 1)
        predict_same_with_label = y_q * predict_sum_binary
        
        predict = sum(predict_same_with_label)
        label = sum(y_q)
        results.append(predict / label)
    return np.array(results).mean()