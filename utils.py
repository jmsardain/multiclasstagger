import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from models import *
from torch_geometric.nn import aggr
import pickle
import torch.nn.functional as F
import math


def train(loader, model, device, optimizer):
    model.train()
    loss_all = 0
    for idx, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)

        labels = torch.tensor(data.y, dtype=torch.float).to(device)
        labels = torch.reshape(labels, (int(list(labels.shape)[0]),1))
        ww = torch.tensor(data.weights, dtype=torch.float).to(device)
        ww = torch.reshape(ww, (int(list(labels.shape)[0]),1))

        loss1 = torch.nn.CrossEntropyLoss(weight=ww)(out, labels)
        loss = loss1
        loss.backward()
        optimizer.step()
        loss_all += loss.item()
    return loss_all


def validate(loader, model, device ):
    model.eval()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        out = out.view(-1, out.shape[-1])

        labels = torch.tensor(data.y, dtype=torch.float).to(device)
        labels = torch.reshape(labels, (int(list(labels.shape)[0]),1))
        ww = torch.tensor(data.weights, dtype=torch.float).to(device)
        ww = torch.reshape(ww, (int(list(labels.shape)[0]),1))

        loss1 = torch.nn.CrossEntropyLoss(weight=ww)(out, labels)
        loss = loss1
        loss_all += loss.item()
    return loss_all



def plot_ROC_curve(loader, model, device, outdir=''):
    model.eval()
    nodes_out = torch.tensor([])
    labels_test = torch.tensor([])

    for data in loader:
        data = data.to(device)

        out = model(data)
        out = out.view(-1, out.shape[-1])

        labels = torch.tensor(data.y, dtype=torch.float).to(device)
        labels = torch.reshape(labels, (int(list(labels.shape)[0]),1))

        nodes_out = torch.cat((nodes_out.clone().detach(), out.clone().detach().cpu()), 0)
        labels_test = torch.cat((labels_test.clone().detach(), labels.clone().detach().cpu()), 0)

    nodes_out = torch.squeeze(nodes_out)
    labels_test = torch.squeeze(labels_test)

    fpr, tpr, thresholds = roc_curve(labels_test, nodes_out )
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    roc_auc = roc_auc_score(labels_test, nodes_out )
    ax.annotate(f'Area Under Curve {roc_auc:.4f}' , xy=(310, 120), xycoords='axes points',
                size=12, ha='right', va='top',
                bbox=dict(boxstyle='round', fc='w'))
    plt.xlabel("False positive rate", size=14)
    plt.ylabel("True positive rate", size=14)
    plt.savefig(outdir+'_roc_curve.png')
