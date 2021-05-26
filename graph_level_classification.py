import pickle
from collections import defaultdict
import numpy as np
from torch_geometric.data import Data, DataLoader
import torch
from dataLoader import load_word_list
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import os.path as osp
# category = ['alt.atheism', 'sci.space']

# doc_list,y = fetch_20newsgroups(subset='all', categories=category, return_X_y=True)
# vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, stop_words="english")
# tf_idf = vectorizer.fit_transform(doc_list)
# print(f"tf idf shape: {tf_idf.shape}")


def build_doc_graph(document, vocab_size, window_size = 20):
    """function to build document graph
    :param document: list of indexed tokens
    :param vocab_size:
    :param window_size:
    :return:
    """
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

    unique_tokens = list(unique_tokens) # transform to list to ensure the fixed iteration order
    token2pos = {token: i for i, token in enumerate(unique_tokens)}

    rows = []
    cols = []
    data = []

    x = [] # stores the one-hot feature matrix
    for token in unique_tokens:
        temp = [0 for _ in range(vocab_size)]
        temp[token2pos[token]] = 1
        x.append(temp)

    x = torch.LongTensor(x)

    for (i, j) in cooccur:
        v = np.log(cooccur[(i, j)] * n / freq[i] / freq[j])
        if v > 0:
            x,y = token2pos[i], token2pos[j]
            rows.append(x)
            cols.append(y)
            data.append(v)

    edge_list = torch.LongTensor(np.vstack([rows, cols]))
    return Data(x = x, edge_list = edge_list, edge_weight = data)



with open("data/20ng/indexed_tokens_list", "rb") as f:
    corpus = pickle.load(f)


vocab_size = len(load_word_list())

data_list = []
for doc in corpus:
    data_list.append(build_doc_graph(doc, vocab_size))

loader = DataLoader(data_list, batch_size=32)







