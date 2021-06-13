from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix, vstack, hstack
import numpy as np
import pickle
import os
from collections import defaultdict
from dataLoader import load_20ng, load_Ned_company, load_others
import argparse

from helper import check_valid_filename


def build_tf_idf(doc_list, max_df=0.8, min_df = 5, stop_words = "english"):
    # TODO: here I use all data to calculate tf-idf, usually you only use training data, but I saw others are also doing like this, so I will remain as it is
    """create tfidf doc-word matrix
    :param doc_list: list of document
    :param max_df: max document frequency to filter for
    :param min_df: min document occurrence/frequency, see sklearn doc for details
    :return: tf_idf sparse matrix and vectorizer
    """
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words=stop_words)
    tf_idf = vectorizer.fit_transform(doc_list)
    return tf_idf, vectorizer


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



def parseArgument():
    """ parse argument
    :return: dictionary of arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    parser.add_argument('--max_df', type=float, default=0.8)
    parser.add_argument('--min_df',type=int, default=5)
    args = parser.parse_args()
    args = vars(args)
    return args


def load_doc_label(file):
    """ load the document list and labels according to file
    :param file:
    :return: (list of raw text, numeric labels)
    """
    check_valid_filename(file)
    # if file == "20ng":
    #     return load_20ng(**args)
    if file == 'ned_company':
        return load_Ned_company(parent)
    else:
        return load_others(file)


if __name__ == "__main__":
    args = parseArgument()
    print(args)
    parent = os.path.join(os.path.dirname(__file__), "data")
    file = args['file']
    data_dir = os.path.join(parent, file)

    obj = load_doc_label(file)
    doc_list = obj['doc_list']
    y = obj['y']

    tf_idf, vectorizer = build_tf_idf(doc_list, max_df=args['max_df'], min_df=args['min_df']) # create tfidf doc-word matrix
    print(f"tf idf shape: {tf_idf.shape}")

    word_key_map = vectorizer.vocabulary_ # word to key mapping
    print(f"vocabulary size: {len(vectorizer.vocabulary_)}")

    # word_list = build_word_list(word_key_map)
    word_list = vectorizer.get_feature_names()

    indexed_tokens_list = build_indexed_tokens_list(doc_list, word_key_map, vectorizer.build_tokenizer()) # for simplicity, use the sklearn tokenizer

    with open(os.path.join(data_dir, "tf_idf"), 'wb') as f:
        pickle.dump(tf_idf, f)

    with open(os.path.join(data_dir, "y"), 'wb') as f:
        pickle.dump(y, f)
    with open(os.path.join(data_dir, 'word_key_map'), 'wb') as f:
        pickle.dump(vectorizer.vocabulary_, f)
    with open(os.path.join(data_dir, "word_list"), 'wb') as f:
        pickle.dump(word_list, f)
    with open(os.path.join(data_dir, 'indexed_tokens_list'), 'wb') as f:
        pickle.dump(indexed_tokens_list, f)
