"""
This script should be executed after executing the build_adj script
"""
import argparse
from collections import defaultdict
import torch
from torch_geometric.data import Data, InMemoryDataset
from dataLoader import *
import os

from helper import check_valid_filename


def build_doc_graph(document, target, window_size=20, cutoff=0):
    """function to build document graph
    The graph is represented as pg.Data object, the node id can be arbitrary provided they are unique and the features on them correspond to the token they represent
    :param document: list of indexed tokens
    :param target: target label for the document
    :param window_size: sliding window size
    :return: pg.Data object
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
    for token in unique_tokens:
        # features.append(idx_pos_map[token])
        features.append(token)

    features = torch.LongTensor(features).view(-1, 1)

    # compute edge list
    token2nodeid = {token: i for i, token in enumerate(unique_tokens)}
    rows = []
    cols = []
    edge_weight = []
    for (i, j) in cooccur:
        v = np.log(cooccur[(i, j)] * n / freq[i] / freq[j])
        if v > cutoff:
            x, y = token2nodeid[i], token2nodeid[j]
            rows.append(x)
            cols.append(y)
            edge_weight.append(v)

    edge_index = torch.LongTensor(np.vstack([rows, cols]))
    edge_weight = torch.Tensor(edge_weight)
    target = torch.LongTensor([[target]])
    data = Data(x=features, y=target, edge_index=edge_index, edge_weight=edge_weight)
    return data



class MyOwnDataset(InMemoryDataset):
    """This follows the torch.geometric example for creating own dataset"""

    def __init__(self, root, category='train', transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        if category == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[1])

    @property
    def processed_file_names(self):
        return ['train.pt', 'test.pt']

    def process_set(self, dataset):
        meta = load_meta(file)
        train_size = meta['train size']
        if dataset == 'train':
            token_list = indexed_tokens_list[:train_size]
            target = y[:train_size]
        else:
            token_list = indexed_tokens_list[train_size:]
            target = y[train_size:]
        data_list = []
        for i, doc, in enumerate(token_list):
            # if i > 20:
            #     break
            print("process %dth document" % (i + 1))
            g = build_doc_graph(doc, target[i], window_size=args['window'], cutoff=args['cutoff'])
            data_list.append(g)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(
            data_list)  # saving collated data object is much much faster than save list of data!
        return data, slices


    def process(self):
        # Read data into huge `Data` list.
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])



def parseArgument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--cutoff', type=float, default=0)
    args = parser.parse_args()
    args = vars(args)
    return args


if __name__ == "__main__":
    """ This script should be run after running `buildCorpusGraph`
    """
    args = parseArgument()
    print(args)
    file = args['file']
    check_valid_filename(file)

    parent = os.path.join(os.path.dirname(__file__), "data")
    data_dir = os.path.join(parent, file)

    indexed_tokens_list = load_indexed_tokens_list(file)

    y = load_y(file)

    word_list = load_word_list(file)

    vocab_size = len(word_list)

    # idx_pos_map = load_idx_pos_map(file)

    dataset = MyOwnDataset(root=data_dir)  # load the dataset, the processing will only be done once
