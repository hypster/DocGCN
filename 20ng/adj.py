from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix, lil_matrix, vstack, hstack
import numpy as np
import pickle
import os
from collections import defaultdict

parent = os.getcwd()
data_dir = os.path.join(parent, '20ng', 'data')
# load 20NG data
categories = ['alt.atheism', 'sci.space']
data_20ng = fetch_20newsgroups(subset='all', categories=categories)
doc_list = data_20ng.data
y = data_20ng.target

with open(os.path.join(data_dir, '20ng_y'), 'wb') as f:
    pickle.dump(y, f)

# create tfidf doc-word matrix
vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, stop_words="english")
tf_idf = vectorizer.fit_transform(doc_list)
print(f"tf idf shape: {tf_idf.shape}")

# save the vocabulary as word to index mapping
word_key_map = vectorizer.vocabulary_
print(f"vocabulary size: {len(vectorizer.vocabulary_)}")
with open(os.path.join(data_dir, '20ng_word_key_map'), 'wb') as f:
    pickle.dump(vectorizer.vocabulary_, f)

word_list = []
for word in word_key_map:
    word_list.append(word)

with open(os.path.join(data_dir, '20ng_word_list'), 'wb') as f:
    pickle.dump(word_list, f)

tokenize = vectorizer.build_tokenizer()  # use the sklearn tokenizer

# save the indexed tokens
indexed_tokens_list = []
for doc in doc_list:
    doc = doc.lower()
    words = tokenize(doc)
    indexed_tokens = []
    for word in words:
        if word in word_key_map:
            if word in word_key_map:
                indexed_tokens.append(word_key_map[word])
    indexed_tokens_list.append(indexed_tokens)

with open(os.path.join(data_dir, '20ng_indexed_tokens_list'), 'wb') as f:
    pickle.dump(indexed_tokens_list, f)


def build_cooccur_map(word_list, indexed_tokens_list, window_size=15):
    """Build cooccur map.

    Co-occurance matrix, count is based on the sliding window size
    this captures the between doc context

    :param word_list:
    :param indexed_tokens_list:
    :param window_size: sliding window size, which is a hyperparameter
    :return: cooccur map
    """
    size = len(word_list)
    '''to sparse use dict instead'''
    # cooccur_matrix = np.zeros((size, size)) #
    cooccur_map = defaultdict(int)

    for indexed_tokens in indexed_tokens_list:
        for i, center in enumerate(indexed_tokens):
            left = max(i - window_size // 2, 0)
            right = min(i + window_size // 2 + 1, len(indexed_tokens))
            neighbors = indexed_tokens[left:i] + indexed_tokens[i + 1:right]
            appeared = set()
            appeared.add(center)
            for neighbor_idx in neighbors:
                if neighbor_idx not in appeared:  # each sliding window contributes at maximum 1 between word pairs
                    appeared.add(neighbor_idx)
                    cooccur_map[center, neighbor_idx] += 1
                    cooccur_map[neighbor_idx, center] += 1

    return cooccur_map


cooccur_map = build_cooccur_map(word_list, indexed_tokens_list)

indexed_token_freq_map = defaultdict(int)
for indexed_tokens in indexed_tokens_list:
    for indexed_token in indexed_tokens:
        indexed_token_freq_map[indexed_token] += 1

total_word_count = sum(len(doc) for doc in indexed_tokens_list)
print(f"total number of tokens in corpus: {total_word_count}")


def build_pmi_matrix(word_list, cooccur_map, total_word_count):
    """build pmi matrix.
    pmi(pointwise mutual information) matrix contains context between words, each entry is computed as log {p(x,y) / ((p(x) * p(y))}
    :param word_list:
    :param cooccur_map:
    :param total_word_count:
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
                    cooccur_map[(i, j)] * total_word_count / indexed_token_freq_map[i] / indexed_token_freq_map[j])
                if v > 0:  # only includes positive pmi as edge
                    rows.append(i)
                    cols.append(j)
                    data.append(v)

    return coo_matrix((data, (rows, cols)), shape=(size, size))



pmi_matrix = build_pmi_matrix(word_list, cooccur_map, total_word_count)
pmi_matrix = lil_matrix(pmi_matrix)


# adjacency matrix structure:
# top-left: PMI
# top-right: tf_idf_transpose
# bottom_left: tf_idf
# bottom_right: zero matrix, we don't have edge between document in contrast to citation network
tf_idf_T = tf_idf.transpose()
doc_size, vocab_size = tf_idf.shape
top = hstack([pmi_matrix, tf_idf_T])

bottom_right = coo_matrix((doc_size, doc_size))
bottom = hstack([tf_idf, bottom_right])
adj = vstack([top, bottom])
with open(os.path.join(data_dir, '20ng_adj'), 'wb') as f:
    pickle.dump(adj, f)
