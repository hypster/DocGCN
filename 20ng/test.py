import pickle
import numpy as np


with open('vocab', 'rb') as f:
    vocab = pickle.load(f)

with open('pmi', 'rb') as f:
    pmi_matrix = pickle.load(f)

with open('data/word_list', 'rb') as f:
    word_list = pickle.load(f)


def test_pmi():
    idx_car = vocab['car']
    connocated = np.argsort(pmi_matrix[idx_car,:])[::-1]
    for w in connocated[:20]:
        print(word_list[w])

if __name__ == '__main__':
    test_pmi()