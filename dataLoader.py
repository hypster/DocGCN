import os
import pickle
import re

import torch
from sklearn.datasets import fetch_20newsgroups
import numpy as np

data_dir = os.path.join(os.path.dirname(__file__), "data")


def load_20ng(category, **kargs):
    """ Load unshuffled data from 20 news group
    :param category:
    :param kargs:
    :return: the document list, the target labels
    """
    meta = {}
    doc_train, y_train = fetch_20newsgroups(subset='train', categories=category, shuffle=True, return_X_y=True)
    doc_test, y_test = fetch_20newsgroups(subset='test', categories=category, shuffle=True, return_X_y=True)
    meta['train size'] = len(doc_train)
    meta['test size'] = len(doc_test)
    doc_list = np.append(doc_train, doc_test)
    y = np.append(y_train, y_test)
    meta['total size'] = len(doc_list)
    ret = {'doc_list': doc_list, 'y': y, 'meta': meta}
    return ret


def load_Ned_company(parent):
    with open(os.path.join(parent, "data", "ned_company"), 'rb') as f:
        data = pickle.load(f)

    x, labels = list(zip(*data))
    set_label = set(label for label in labels)
    label2target = {label: i for i, label in enumerate(set_label)}
    with open(os.path.join(parent, "ned_company", "label2target"), 'wb') as f:
        pickle.dump(label2target, f)
    y = []
    for label in labels:
        y.append(label2target[label])
    ret = {'doc_list': x, 'y': y}
    return ret

def load_others(file):
    with open(os.path.join(data_dir, file, 'doc_list'), 'rb') as f:
        doc_list = pickle.load(f)

    with open(os.path.join(data_dir, file, 'y'), 'rb') as f:
        y = pickle.load(f)
    ret = {'doc_list': doc_list, 'y': y}
    return ret

def load_meta(file):
    with open(os.path.join(data_dir, file, 'meta'), 'rb') as f:
        ret = pickle.load(f)
    return ret

def load_tf_idf(file):
    with open(os.path.join(data_dir, file, 'tf_idf'), 'rb') as f:
        ret = pickle.load(f)
    return ret

"""
warning: calling this function is deprecated. The original word_list is built using the iteration sequence from sklearn.vectorizer.vocabulary_ attribute. Sklearn vectorizer also provides get_feature_names() method which gives us the word list which is the inverse mapping of vectorizer.vocabulary_ attribute. I overlooked this utility function, so we need a middle mapping to transform from the token index(as saved in sklearn vectorizer) to the positional index in word_list (which is separately created by me). The new implementation will use the internal function of library.
"""


def load_idx_pos_map(dataset):
    """ deprecated code to transform from token to positional index in the word_list
    :param dataset:
    :return:
    """
    m = load_word_key_map(dataset)  # {word -> key}
    wl = load_word_list(dataset)  # [word...]
    return {m[w]: i for i, w in enumerate(wl)}


def load_word_key_map(dataset):
    with open(os.path.join(data_dir, dataset, 'word_key_map'), 'rb') as f:
        m = pickle.load(f)
    return m


def load_y(dataset):
    with open(os.path.join(data_dir, dataset, 'y'), 'rb') as f:
        y = pickle.load(f)
    return y

def load_class_names(dataset):
    with open(os.path.join(data_dir, dataset, 'label_list'), 'rb') as f:
        names = pickle.load(f)
    return names

def load_label_target_map(dataset):
    with open(os.path.join(data_dir, dataset, 'label2target'), 'rb') as f:
        m = pickle.load(f)
    return m



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

def get_train_size(dataset):
    train_size = None
    if dataset == "20ng":
        train_size = get_20ng_train_size()
    else:
        with open(os.path.join(data_dir, dataset, 'meta'), 'rb') as f:
            meta = pickle.load(f)
            train_size = meta['train size']
    return train_size

def load_edge_index_weight(dataset):
    """ load edge index
    transform sparse adjacency matrix and transform into pG edge index format with dimension (2, *)
    :return: edge_index, edge_weight
    """
    with open(os.path.join(data_dir, dataset, 'adj'), 'rb') as f:
        adj = pickle.load(f)
    # for i in range(k - 1):  # matrix power to produce homogeneous adjacency matrix
    #     adj = adj * adj

    # vocab_size = len(load_word_list(dataset))
    # adj = adj[vocab_size:, vocab_size:]
    # adj = normalize(adj, axis=0, norm='l1') # normalize to column stochastic matrix
    # mat = coo_matrix(adj.todense() > 0.01, shape=adj.shape) # transform to 0-1 matrix， 0.01 is chosen so the resulting matrix is sparse and approximates a more standard network

    edge_index = np.vstack([adj.row, adj.col])  # transform to edge index array
    edge_weight = adj.data
    edge_index = torch.LongTensor(edge_index)
    edge_weight = torch.FloatTensor(edge_weight)
    return edge_index, edge_weight

def load_doc_list(dataset):
    with open(os.path.join(data_dir, dataset, 'doc_list'), 'rb') as f:
        doc_list = pickle.load(f)
    return doc_list


def extract_log(file):
    reg = re.compile('[\d]*[.][\d]+')
    loss = []
    train_acc = []
    val_acc = []
    test_acc = []
    f1_score = []
    arr = []
    with open(file, "r") as f:
        lines = f.readlines()
        for l in lines[:-1]:
            temp = []
            for i, obj in enumerate(reg.finditer(l)):
                temp.append(float(obj[0]))

            arr.append(temp)

    loss, train_acc, val_acc, test_acc, f1_score = list(zip(*arr))


    return {'loss': loss, 'train acc': train_acc, 'val acc': val_acc, 'test acc': test_acc, 'f1 score': f1_score}