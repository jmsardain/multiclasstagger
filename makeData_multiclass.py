import numpy as np
import pandas as pd
# import torch
import uproot

# from utils import *
# from models import *
import os
import gc

import torch
from torch_geometric.data import Data
from torch.utils.data import TensorDataset, DataLoader
import pickle
import networkx as nx
from sklearn.model_selection import train_test_split
# from utils import *
# from models import *
import h5py
import glob

def normalize(x, dic, feature):
    mean, std = np.mean(x), np.std(x)
    out =  (x - mean) / std
    dic[str(feature)] = [mean, std]
    return out, mean, std

def to_networkx(data):
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    edges = data.edge_index.t().tolist()
    G.add_edges_from(edges)
    return G

def apply_save_log(x):

    #########
    epsilon = 1e-10
    #########

    minimum = x.min()
    if x.min() <= 0:
        x = x - x.min() + epsilon
    else:
        minimum = 0
        epsilon = 0

    return np.log(x), minimum, epsilon

def get_graphs(ijet, file):
    labels = file['jets'][ijet:ijet+1]['R10TruthLabel_R22v1']
    EFlow = file['flow'][ijet:ijet+1]['flow_energy']
    ptFlow = file['flow'][ijet:ijet+1]['flow_pt']
    detaFlow = file['flow'][ijet:ijet+1]['flow_deta']
    dphiFlow = file['flow'][ijet:ijet+1]['flow_dphi']
    drFlow = file['flow'][ijet:ijet+1]['flow_dr']

    ptJet = file['jets'][ijet:ijet+1]['pt']
    EJet = file['jets'][ijet:ijet+1]['energy']
    ## when changing the labels values, order counts, since one label might be reset to another label
    if labels == 10: labels = 0 ## qcd
    if labels == 1:  labels = 1 ## tqqb
    if labels == 2:  labels = 2 ## Wqq
    if labels == 6:  labels = 2 ## Wqq_From_t
    if labels == 3:  labels = 3 ## Zbb
    if labels == 4:  labels = 3 ## Zcc
    if labels == 5:  labels = 3 ## Zqq
    if labels == 11: labels = 4 ## Hbb
    if labels == 12: labels = 4 ## Hcc
    if labels == 13: labels = 4 ## other from H

    flow_deta_filtered = detaFlow[~np.isnan(detaFlow)]
    flow_dphi_filtered = dphiFlow[~np.isnan(dphiFlow)]
    flow_dr_filtered   = drFlow[~np.isnan(drFlow)]

    flow_pt_filtered     = ptFlow[~np.isnan(ptFlow)]
    flow_energy_filtered = EFlow[~np.isnan(EFlow)]

    jet_energy_filtered = EJet[~np.isnan(EJet)]
    jet_pt_filtered     = ptJet[~np.isnan(ptJet)]

    flow_pt_filtered_log     = np.log(flow_pt_filtered)
    flow_energy_filtered_log = np.log(flow_energy_filtered)

    frac_pt_filtered = np.log( np.array(flow_pt_filtered)     / np.array(jet_pt_filtered)     )
    frac_E_filtered  = np.log( np.array(flow_energy_filtered) / np.array(jet_energy_filtered) )


    x = torch.tensor([flow_deta_filtered, flow_dphi_filtered, flow_pt_filtered_log, frac_pt_filtered, flow_energy_filtered_log, frac_E_filtered, flow_dr_filtered], dtype=torch.float).t()
    edge_index = torch.combinations(torch.arange(len(flow_pt_filtered)), 2).t().contiguous()

    graph = Data(x=x, edge_index=edge_index, y=torch.tensor(labels, dtype=torch.float))

    return graph

# Main function.
def main():
    file_list = glob.glob("/data/jmsardain/MultiClassTagger/h5files/*_000001.output.h5")


    graph_list = []
    for idx, file_path in enumerate(file_list):
        print("Running on file {}: {}".format(idx, file_path))
        with h5py.File(file_path, 'r') as file:
            Njets = len(file['jets'])
            print("Number of jets : {}".format(Njets))

            print("I am here")

            for ijet in range(Njets-1):
                if ijet%1000==0:
                    print("jet #{}".format(ijet))
                graph_list.append(get_graphs(ijet, file))

                # if ijet==0:
                #     networkx_graph = to_networkx(graph)
                #     file_path = "graph.graphml"
                #     nx.write_graphml(networkx_graph, file_path)

    output_path_graphs = "data/graphs"
    size_train = 0.80
    graphs_test, graph_train = train_test_split(graph_list, test_size = size_train, random_state = 144)
    torch.save(graph_train, output_path_graphs + "_train.pt")
    torch.save(graphs_test, output_path_graphs + "_test.pt")


# Main function call.
if __name__ == '__main__':
    main()
    pass
