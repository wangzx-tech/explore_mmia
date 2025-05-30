import torch
from sklearn import metrics
from torch import nn
import numpy as np


def ACC(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.accuracy_score(y_true, y_pred)


def Cohen_Kappa(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.cohen_kappa_score(y_true, y_pred)


def F1_score(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.f1_score(y_true, y_pred, average='macro', zero_division=0.)


def Recall(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.recall_score(y_true, y_pred, average='macro', zero_division=0.)


def Precision(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.precision_score(y_true, y_pred, average='macro', zero_division=0.)


def cls_report(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.classification_report(y_true, y_pred, digits=4, zero_division=0.)


def confusion_matrix(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.confusion_matrix(y_true, y_pred)


def feature_loss(feature):
    bs, phase, embed_dim = feature.shape
    # For share feature: usage mean as gt
    feature_gt = torch.mean(feature, dim=1, keepdim=True)
    loss_func = torch.nn.MSELoss(reduction='sum')
    loss_share = loss_func(feature_gt.repeat(1, phase, 1), feature)
    loss = loss_share / phase
    return loss
