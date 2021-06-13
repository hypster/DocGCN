import argparse
import os
import pickle
from collections import defaultdict
import numpy as np
from scipy.sparse import coo_matrix, hstack, vstack
import dataLoader


def build_pmi_matrix(word_list, cooccur_map, token_freq_map, total_word_count, cutoff = 0, **kwargs):
    """build pmi matrix.
    pmi(pointwise mutual information) matrix contains context between words, each entry is computed as log {p(x,y) / ((p(x) * p(y))}
    :param word_list:
    :param cooccur_map:
    :param total_word_count:
    :param cutoff: the cutoff value for deciding the creation of edge
    :return: pmi_matrix
    """
    size = len(word_list)
    rows = []
    cols = []
    data = []

    for i in range(size):
        for j in range(size):
            if (i, j) in cooccur_map:
                v = np.log(
                    cooccur_map[(i, j)] * total_word_count / token_freq_map[i] / token_freq_map[j])
                if v > cutoff:  # only includes positive pmi as edge
                    rows.append(i)
                    cols.append(j)
                    data.append(v)

    return coo_matrix((data, (rows, cols)), shape=(size, size))


def build_corpus_graph(word_list, tf_idf, indexed_tokens_list, window = 5, cutoff = 0):
    total_word_count = sum(len(doc) for doc in indexed_tokens_list)
    print(f"total number of tokens in corpus: {total_word_count}")
    cooccur_map, token_freq_map = build_cooccur_map(indexed_tokens_list, window_size=window)
    pmi_matrix = build_pmi_matrix(word_list, cooccur_map, token_freq_map, total_word_count, cutoff)
    adj = build_hetereogenous_graph(tf_idf, pmi_matrix)
    return adj


def build_hetereogenous_graph(tf_idf, pmi_matrix):
    """ build hetereogenous graph containing document and word nodes
    adjacency matrix structure:
    top-left: PMI
    top-right: tf_idf_transpose
    bottom_left: tf_idf
    bottom_right: zero matrix, we don't have edge between document
        """

    tf_idf_T = tf_idf.transpose()
    doc_size, vocab_size = tf_idf.shape
    top = hstack([pmi_matrix, tf_idf_T])

    bottom_right = coo_matrix((doc_size, doc_size))
    bottom = hstack([tf_idf, bottom_right])
    adj = vstack([top, bottom])
    return adj


def build_cooccur_map(indexed_tokens_list, window_size=20):
    """Build cooccur map.
    Co-occurance matrix, count is based on the sliding window size
    this captures the between doc context
    :param indexed_tokens_list:
    :param window_size: sliding window size
    :return: cooccur map: (word_i, word_j) -> count, token_freq_map: (word_i) -> count
    """
    cooccur_map = defaultdict(int)
    token_freq_map = defaultdict(int)
    for indexed_tokens in indexed_tokens_list:
        for i, center in enumerate(indexed_tokens):
            left = max(i - window_size // 2, 0)
            right = min(i + window_size // 2 + 1, len(indexed_tokens))
            neighbors = indexed_tokens[left:i] + indexed_tokens[i + 1:right]
            pair_set = set()
            token_freq_map[center] += 1
            for neighbor in neighbors:
                if (neighbor, center) in pair_set or (center, neighbor) in pair_set: # if same pair more than twice in the same sliding window, we skip the count
                    continue

                pair_set.add((neighbor, center))
                pair_set.add((center, neighbor))
                cooccur_map[center, neighbor] += 1
                cooccur_map[neighbor, center] += 1

    return cooccur_map, token_freq_map


def parseArgument():
    """ parse argument
    :return: dictionary of arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--cutoff',type = float, default=0)
    args = parser.parse_args()
    args = vars(args)
    return args

if __name__ == "__main__":
    args = parseArgument()
    print(args)
    parent = os.path.join(os.path.dirname(__file__), "data")
    file = args['file']
    data_dir = os.path.join(parent, file)

    word_list = dataLoader.load_word_list(file)
    indexed_tokens_list = dataLoader.load_indexed_tokens_list(file)
    tf_idf = dataLoader.load_tf_idf(file)

    adj = build_corpus_graph(word_list, tf_idf, indexed_tokens_list, window = args['window'], cutoff=args['cutoff'])

    with open(os.path.join(data_dir, 'adj'), 'wb') as f:
        pickle.dump(adj, f)
