import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import comb
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, adjusted_rand_score
import torch
from typing import Optional

def rand_index_np(classes, clusters):
    classes = classes.astype(np.int32)
    clusters = clusters.astype(np.int32)
    # scipy.special.comb(n,k):  C_n^k
    # np.bincount: calculate each sample size of ordered-clusters 
    # e.g. np.bincount([1, 2, 1, 3, 4, 4])=[2,1,1,2], 
    # i.e. there are 4 clusters with samples of 2,1,1, and 2, respectively
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum([comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum() for i in set(clusters)])
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    randi=(tp + tn) / (tp + fp + fn + tn)
    return randi


def acc_np(labels, preds):
    """
    Compute the confusion matrix and accuracy corresponding to the best cluster-to-class assignment.

    :param labels: Label array
    :type labels: np.array
    :param pred: Predictions array
    :type pred: np.array
    :return: Accuracy and confusion matrix
    :rtype: Tuple[float, np.array]
    """
    cmat = confusion_matrix(labels, preds)
    ri, ci = linear_sum_assignment(-cmat)
    ordered = cmat[np.ix_(ri, ci)]
    acc = np.sum(np.diag(ordered))/np.sum(ordered)
    # return acc, ordered
    return acc

def nmi_np(labels, preds):
    return(normalized_mutual_info_score(labels, preds))

def ari_np(labels, preds):
    return(adjusted_rand_score(labels, preds))


def cluster_accuracy(y_true, y_predicted, cluster_number: Optional[int] = None):
    """
    Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
    determine reassignments.

    :param y_true: list of true cluster numbers, an integer array 0-indexed
    :param y_predicted: list  of predicted cluster numbers, an integer array 0-indexed
    :param cluster_number: number of clusters, if None then calculated from input
    :return: reassignment dictionary, clustering accuracy
    """
    if cluster_number is None:
        cluster_number = (
            max(y_predicted.max(), y_true.max()) + 1
        )  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
    return reassignment, accuracy


class Metric(torch.nn.Module):
    def __init__(self, name='clustering_metrics', keys=[]) -> None: 
        super(Metric, self).__init__() 
        self.keys = keys
        self.name = name
        self.training = False

    def forward(self, labels, preds, type='numpy', decimals=4):   
        if self.keys=={}:
            return {}
        else:
            if type=='numpy':
                if isinstance(labels, torch.Tensor):
                    labels = labels.detach().cpu().numpy()
                if isinstance(preds, torch.Tensor):
                    preds = preds.detach().cpu().numpy()       

            return {key: round(globals()[key if type=='torch' else key+'_np'](labels, 
                    preds), decimals) for key in self.keys} if len(self.keys)>0 else {}