import os
import pickle

if __name__ == "__main__":
    parent = os.path.dirname(__file__)
    data_dir = os.path.join(parent, "data", "r8")

""" train and test data follows in order, so we just save the whole doc list and train size instead of split between train and test. This is also to conform to previous written interface in build graph script.
"""
with open(os.path.join(data_dir, "id_set_label.txt"), 'r') as f:
    train_size = 0
    labels = []
    for i, line in enumerate(f):
        arr = [e.strip() for e in line.split('\t')]
        train_test_type = arr[1]
        label = arr[2]
        if train_test_type == 'train':
            train_size += 1
        labels.append(label)

doc_list = []

meta = {'train size': train_size}
meta['total size': len(labels)]
meta['test size'] = len(labels) - train_size

label = set(labels)
label_list = list(label)
label2target = {l: i for i, l in enumerate(label_list)}

y = [label2target[label] for label in labels]

with open(os.path.join(data_dir, "corpus.txt"), 'r') as f:
    for document in f:
        doc_list.append(document)

with open(os.path.join(data_dir, "meta"), 'r') as f:
    pickle.dump(meta, f)

with open(os.path.join(data_dir, "label2target"), 'wb') as f:
    pickle.dump(label2target, f)

with open(os.path.join(data_dir, "y"), 'wb') as f:
    pickle.dump(y, f)

with open(os.path.join(data_dir, "label_list"), 'wb') as f:
    pickle.dump(label_list, f)


with open(os.path.join(data_dir, "doc_list"), 'wb') as f:
    pickle.dump(doc_list, f)
