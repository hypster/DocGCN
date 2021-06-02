from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix, vstack, hstack
import numpy as np
import pickle
import os
from collections import defaultdict
from dataLoader import load_20ng, load_Ned_company
import argparse

from helper import check_valid_filename


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


def build_tf_idf(doc_list, max_df=0.8, min_df=0.5):
    """create tfidf doc-word matrix
    :param doc_list: list of document
    :param max_df: max document frequency to filter for
    :param min_df: min document occurrence/frequency, see sklearn doc for details
    :return: tf_idf sparse matrix and vectorizer
    """
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words="english")
    tf_idf = vectorizer.fit_transform(doc_list)
    return tf_idf, vectorizer


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


def build_hetereogenous_graph(tf_idf, pmi_matrix):
    """ build hetereogenous graph containing document and word nodes
    adjacency matrix structure:
    top-left: PMI
    top-right: tf_idf_transpose
    bottom_left: tf_idf
    bottom_right: zero matrix, we don't have edge between document in contrast to citation network
        """

    tf_idf_T = tf_idf.transpose()
    doc_size, vocab_size = tf_idf.shape
    top = hstack([pmi_matrix, tf_idf_T])

    bottom_right = coo_matrix((doc_size, doc_size))
    bottom = hstack([tf_idf, bottom_right])
    adj = vstack([top, bottom])
    return adj


def build_indexed_tokens_list(doc_list, word_key_map, tokenize):
    """ transform the raw text into list of indexed token
    :param doc_list: list of raw text
    :param word_key_map: word to key map
    :param tokenize: tokenize function
    :return: a list of indexed token
    """
    indexed_tokens_list = []
    for doc in doc_list:
        doc = doc.lower()
        words = tokenize(doc)
        indexed_tokens = []
        for word in words:
            if word in word_key_map:  # keep only words in the vocabulary
                indexed_tokens.append(word_key_map[word])
        indexed_tokens_list.append(indexed_tokens)
    return indexed_tokens_list

"""deprecated function, the sklearn vectorizer provides the utility function get_feature_names() to do the same thing and is the inverse mapping of _vocabulary attribute, if we use this function, the word_list order is different from the utility function, which means we need a third mapping to transform from token to positional index in the word_list
"""
def build_word_list(word_key_map):
    """ build word list
    the order in the list will be used as unique identity for each word
    :param word_key_map:
    :return: list of words
    """
    word_list = []
    for word in word_key_map:
        word_list.append(word)
    return word_list


def parseArgument():
    """ parse argument
    :return: dictionary of arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    # subcategory = ['alt.atheism', 'sci.space']
    parser.add_argument('--category', nargs='+', default=None)
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--max_df', type=float, default=0.8)
    parser.add_argument('--min_df',type=int, default=5)
    parser.add_argument('--cutoff',type = float, default=0)
    args = parser.parse_args()
    args = vars(args)
    return args


def load_doc_label(file):
    """ load the document list and labels according to file
    :param file:
    :return: (list of raw text, numeric labels)
    """
    check_valid_filename(file)
    if file == "20ng":
        return load_20ng(**args)
    elif file == 'ned_company':
        return load_Ned_company(parent)


if __name__ == "__main__":
    args = parseArgument()
    print(args)
    parent = os.path.join(os.path.dirname(__file__), "data")
    file = args['file']
    data_dir = os.path.join(parent, file)

    doc_list, y = load_doc_label(file)

    tf_idf, vectorizer = build_tf_idf(doc_list, max_df=args['max_df'], min_df=args['min_df']) # create tfidf doc-word matrix
    print(f"tf idf shape: {tf_idf.shape}")

    word_key_map = vectorizer.vocabulary_ # word to key mapping
    print(f"vocabulary size: {len(vectorizer.vocabulary_)}")

    # word_list = build_word_list(word_key_map)
    word_list = vectorizer.get_feature_names()


    indexed_tokens_list = build_indexed_tokens_list(doc_list, word_key_map, vectorizer.build_tokenizer()) # for simplicity, use the sklearn tokenizer

    cooccur_map, token_freq_map = build_cooccur_map(indexed_tokens_list, window_size=args['window'])

    total_word_count = sum(len(doc) for doc in indexed_tokens_list)
    print(f"total number of tokens in corpus: {total_word_count}")

    pmi_matrix = build_pmi_matrix(word_list, cooccur_map, token_freq_map, total_word_count, args['cutoff'])

    adj = build_hetereogenous_graph(tf_idf, pmi_matrix)

    with open(os.path.join(data_dir, "y"), 'wb') as f:
        pickle.dump(y, f)
    with open(os.path.join(data_dir, 'word_key_map'), 'wb') as f:
        pickle.dump(vectorizer.vocabulary_, f)
    with open(os.path.join(data_dir, "word_list"), 'wb') as f:
        pickle.dump(word_list, f)
    with open(os.path.join(data_dir, 'indexed_tokens_list'), 'wb') as f:
        pickle.dump(indexed_tokens_list, f)
    # with open(os.path.join(data_dir, "pmi"), "wb") as f:
    #     pickle.dump(pmi_matrix, f)
    with open(os.path.join(data_dir, 'adj'), 'wb') as f:
        pickle.dump(adj, f)
