import os
import pickle
from sklearn.datasets import fetch_20newsgroups

import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch_geometric import nn
from torch_geometric.data import Data
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, coo_matrix

data_dir = os.path.join(os.path.dirname(__file__), "data")

def load_20ng(category, **kargs):
    if category == ['all']:
        category = None
    data_20ng = fetch_20newsgroups(subset='all', categories=category)
    doc_list = data_20ng.data
    y = data_20ng.target
    return doc_list, y



def load_Ned_company(parent):
    with open(os.path.join(parent, "data", "ned_company"), 'rb') as f:
        data = pickle.load(f)

    x,labels = list(zip(*data))
    set_label = set(label for label in labels)
    label2target = {label: i for i, label in enumerate(set_label)}
    with open(os.path.join(parent, "ned_company", "label2target"), 'wb') as f:
        pickle.dump(label2target, f)
    y = []
    for label in labels:
        y.append(label2target[label])
    return x,y

def load_idx_pos_map(dataset):
    m = load_word_key_map(dataset)
    wl = load_word_list(dataset)
    return {m[w]: i for i, w in enumerate(wl)}

def load_word_key_map(dataset):
    with open(os.path.join(data_dir, dataset, 'word_key_map'), 'rb') as f:
        m = pickle.load(f)
    return m

def load_labels(dataset):
    with open(os.path.join(data_dir, dataset, 'y'), 'rb') as f:
        y = pickle.load(f)
    return y

def load_word_list(dataset):
    with open(os.path.join(data_dir, dataset, 'word_list'), 'rb') as f:
        word_list = pickle.load(f)
    return word_list

def load_adj(dataset):
    with open(os.path.join(data_dir, dataset, 'adj'), 'rb') as f:
        adj = pickle.load(f)
    return adj

# def load_pmi_matrix(dataset):
#     with open(os.path.join(data_dir, dataset, 'pmi'), 'rb') as f:
#         pmi = pickle.load(f)
#     return pmi

def load_graph_list(dataset):
    with open(os.path.join(data_dir, dataset, 'ind_graph'), 'rb') as f:
        return pickle.load(f)

def load_indexed_tokens_list(dataset):
    with open(os.path.join(data_dir, dataset, 'indexed_tokens_list'), 'rb') as f:
        return pickle.load(f)


def get_20ng_train_size():
    """the sklearn function only has train and test split, we use train set to split between train and validation set
    :return:
    """
    return fetch_20newsgroups(subset='train').target.shape[0]

def load_edge_index_weight(dataset, k=1):
    """ load adjacency matrix and transform into edge_index
    :return: edge_index, dtype torch.long
    """
    with open(os.path.join(data_dir, dataset, 'adj'), 'rb') as f:
        adj = pickle.load(f)
    # matrix power to produce homogeneous adjacency matrix
    for i in range(k-1):
        adj = adj * adj

    vocab_size = len(load_word_list(dataset))
    # adj = adj[vocab_size:, vocab_size:]

    # adj = normalize(adj, axis=0, norm='l1') # normalize to column stochastic matrix

    # mat = coo_matrix(adj.todense() > 0.01, shape=adj.shape) # transform to 0-1 matrix， 0.01 is chosen so the resulting matrix is sparse and approximates a more standard network

    edge_index = np.vstack([adj.row, adj.col]) # transform to edge index array
    edge_weight = adj.data

    return edge_index, edge_weight

    # i = torch.LongTensor(np.vstack((adj.row, adj.col)))
    # v = torch.FloatTensor(adj.data)
    # adj = torch.sparse.Tensor(i, v, adj.shape, type=torch.long)
    # return adj

if __name__ == "__main__":

    edge_index, edge_weight = load_edge_index_weight("20ng")
    print("finish")
    # xs = []

    # ys = []
    # values = []
    # for i, j in adj.indices().T:
    #     if adj[i][j] > 0.4:
    #       xs.append(i)
    #       ys.append(j)
    #       values.append(adj[i][j])
    #
    # adj = torch.sparse.FloatTensor(np.vstack([xs, ys]), values, adj.shape)

    # adj = torch.mm(adj, adj)
    #

    # vocab_size = len(load_word_list())
    # x = torch.ones(size=(adj.shape[0], 20))
    # data = Data(x=x, edge_index=adj)






