import matplotlib.pyplot as plt
import argparse
import sys
import math
import numpy as np
import pandas as pd
import datetime, os
import torch
import yaml
from torch_geometric.data import DataListLoader, DataLoader
from torch_geometric.utils import degree

from utils import *
from models import *

def load_yaml(file_name):
    assert(os.path.exists(file_name))
    with open(file_name) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


# Main function.
def main():

    parser = argparse.ArgumentParser(description='Train with configurations')
    add_arg = parser.add_argument
    add_arg('config', help="job configuration")
    args = parser.parse_args()
    config_file = args.config
    config = load_yaml(config_file)

    path_to_train = config['data']['path_to_train']
    path_to_test = config['data']['path_to_test']

    graph_list_train = torch.load(path_to_train)
    graph_list_test  = torch.load(path_to_test)

    size_train = config['data']['size_train']
    graph_list_val = graph_list_train[int(len(graph_list_train)*size_train) : int(len(graph_list_train))]
    graph_list_train = graph_list_train[0 : int(len(graph_list_train)*size_train) ]


    print('Prepare model')
    choose_model = config['architecture']['choose_model']

    ## num_classes = 5 (QCD, top, W, Z, H)
    if choose_model == "PFN":
        ## features are : deta, dphi, lnpT , ln pT / sum pT , lnE, ln E/ sum E, deltaR (flow and jet)
        model = ParticleFlowNetwork(input_dims=7, num_classes=5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print('Prepare optimizer')
    ## define optimizer
    learning_ratio = config['architecture']['learning_rate']
    optimizer1 = optim.Adam(model.parameters(), lr=learning_ratio)

    print('Prepare DataLoaders')
    ## create DataLoaders
    batch_size = config['architecture']['batch_size']
    dataloader_train = DataLoader(graph_list_train, batch_size=batch_size, shuffle=True)
    dataloader_val   = DataLoader(graph_list_val,   batch_size=256, shuffle=True)
    dataloader_test  = DataLoader(graph_list_test,  batch_size=32,  shuffle=True)

    print('Start training')
    train_loss = []
    val_loss   = []
    n_epochs  = config['architecture']['n_epochs']
    path_to_save = config['data']['path_to_save']
    model_name = config['data']['model_name']

    metrics_filename = path_to_save+"/losses.txt"

    for epoch in range(n_epochs):
        print("Epoch:{}".format(epoch+1))

        optimizer = optimizer1

        train_loss.append(train(dataloader_train, model, device, optimizer))
        val_loss.append(validate(dataloader_val, model, device))

        print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f},'.format(epoch, train_loss[epoch], val_loss[epoch]))
        torch.save(model.state_dict(), path_to_save+model_name+"e{:03d}".format(epoch+1) + "_losstrain{:.3f}".format(train_loss[epoch]) + "_lossval{:.3f}".format(val_loss[epoch]) + ".pt")

    metrics = pd.DataFrame({"Train_Loss":train_loss,"Val_Loss":val_loss})
    metrics.to_csv(metrics_filename, index = False)

    ## plot ROC curve
    plot_ROC_curve(dataloader_val, model, device, outdir=path_to_save+model_name)

    fig, ax = plt.subplots()
    ax.plot(train_loss, label='loss')
    ax.plot(val_loss, label='val_loss')
    ax.set_xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(path_to_save + '/Losses_train_pumitigation.png')
    plt.clf()

    return

# Main function call.
if __name__ == '__main__':
    main()
    pass
