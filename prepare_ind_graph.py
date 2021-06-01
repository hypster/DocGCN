"""
This script should be executed after executing the build_adj script
"""
import argparse
import pickle
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from scipy.sparse import coo_matrix

import numpy as np
from torch_geometric.data import Data, DataLoader, Dataset, InMemoryDataset
import torch
from sklearn.datasets import fetch_20newsgroups
from dataLoader import *
import os


def build_doc_graph(document, target, idx_pos_map, vocab_size, window_size=20):
    """function to build document graph
    :param document: list of indexed tokens
    :param target:
    :param idx_pos_map: type `map` {token index: pos in the word list}
    :param vocab_size:
    :param window_size:
    :return:
    """
    # compute co-occur frequencies
    freq = defaultdict(int)
    cooccur = defaultdict(int)
    unique_tokens = set()
    n = len(document)
    for i, center in enumerate(document):
        unique_tokens.add(center)
        left = max(i - window_size // 2, 0)
        right = min(i + window_size // 2 + 1, n)
        freq[center] += 1
        neighbors = document[left:i] + document[i + 1:right]
        pair_set = set()
        for neighbor in neighbors:
            if (neighbor, center) in pair_set or (center, neighbor) in pair_set:
                continue
            pair_set.add((neighbor, center))
            pair_set.add((center, neighbor))
            cooccur[(center, neighbor)] += 1
            cooccur[(neighbor, center)] += 1

    n = len(unique_tokens)

    unique_tokens = list(unique_tokens)  # transform to list to ensure the fixed iteration order

    features = []
    for i, token in enumerate(unique_tokens):
        features.append(idx_pos_map[token])

    features = torch.LongTensor(features).view(-1,1)

    # compute edge list
    token2nodeid = {token: i for i, token in enumerate(unique_tokens)}
    rows = []
    cols = []
    data = []
    for (i, j) in cooccur:
        v = np.log(cooccur[(i, j)] * n / freq[i] / freq[j])
        if v > 0:
            x, y = token2nodeid[i], token2nodeid[j]
            rows.append(x)
            cols.append(y)
            data.append(v)

    edge_list = torch.LongTensor(np.vstack([rows, cols]))
    data = torch.Tensor(data)
    target = torch.LongTensor([[target]])
    data = Data(x=features, y=target, edge_index=edge_list, edge_weight=data)
    return data


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        for i, doc, in enumerate(indexed_tokens_list):
            # if i > 20:
            #     break
            print("process %dth" % (i + 1))
            data_list.append(build_doc_graph(doc, y[i], idx_pos_map, vocab_size=vocab_size))


        for g in data_list:
            g.y = torch.LongTensor([[g.y]])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    args = parser.parse_args()
    args = vars(args)

    file = args['file']
    if file not in ("20ng", "ned_company"):
        print("not supported dataset")
        exit()

    parent = os.path.join(os.path.dirname(__file__), "data")
    data_dir = os.path.join(parent, file)
    with open(os.path.join(data_dir, "indexed_tokens_list"), "rb") as f:
        indexed_tokens_list = pickle.load(f)

    with open(os.path.join(data_dir, "y"), "rb") as f:
        y = pickle.load(f)

    with open(os.path.join(data_dir, "word_list"), "rb") as f:
        word_list = pickle.load(f)

    vocab_size = len(word_list)
    idx_pos_map = load_idx_pos_map(file)

    dataset = MyOwnDataset(root = os.path.join("data", file)) # load the dataset, the processing will only be done once








