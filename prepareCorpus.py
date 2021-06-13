import argparse
import os
import pickle
import random
from helper import check_valid_filename


# def prepare_r8(data_dir):
#     """ train and test data follows in order, so we just save the whole doc list and train size instead of split between train and test. This is also to conform to previous written interface in build graph script.
#     """
#     meta = {}
#     with open(os.path.join(data_dir, "id_set_label.txt"), 'r') as f:
#         train_size = 0
#         labels = []
#         for i, line in enumerate(f):
#             arr = [e.strip() for e in line.split('\t')]
#             train_test_type = arr[1]
#             label = arr[2]
#             if train_test_type == 'train':
#                 train_size += 1
#             labels.append(label)
#
#     doc_list = []
#
#     meta['train size'] = train_size
#     meta['total size'] = len(labels)
#     meta['test size'] = len(labels) - train_size
#
#     label = set(labels)
#     label_list = list(label)
#     label2target = {l: i for i, l in enumerate(label_list)}
#
#     y = [label2target[label] for label in labels]
#
#     with open(os.path.join(data_dir, "corpus.txt"), 'r') as f:
#         for document in f:
#             doc_list.append(document)
#
#     with open(os.path.join(data_dir, "meta"), 'wb') as f:
#         pickle.dump(meta, f)
#
#     with open(os.path.join(data_dir, "label2target"), 'wb') as f:
#         pickle.dump(label2target, f)
#
#     with open(os.path.join(data_dir, "y"), 'wb') as f:
#         pickle.dump(y, f)
#
#     with open(os.path.join(data_dir, "label_list"), 'wb') as f:
#         pickle.dump(label_list, f)
#
#     with open(os.path.join(data_dir, "doc_list"), 'wb') as f:
#         pickle.dump(doc_list, f)


def parseArgument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    args = parser.parse_args()
    args = vars(args)
    return args

def get_doc_list_y(parent):
    doc_list = []
    y_list = []
    for y, _class in enumerate(os.listdir(parent)):
        p = os.path.join(parent, _class)
        for file in os.listdir(p):
            with open(os.path.join(p, file), 'r') as f:
                doc_list.append(f.read())
                y_list.append(y)
    return doc_list, y_list

def get_class_name(train_dir):
    classnames = []
    for y, _class in enumerate(os.listdir(train_dir)):
        classnames.append(_class)
    return classnames


def shuffle_doc_y_together(doc_list, y):
    l = list(zip(doc_list, y))
    random.shuffle(l)
    doc_list, y = zip(*l)
    return doc_list, y


def prepare(corpus_dir, data_dir, file):
    train_doc_list = []
    test_doc_list = []
    y_train = []
    y_test = []
    doc_list = []
    label2target = {}
    y = []
    meta = {}
    train_id_list = []
    test_id_list = []
    with open(os.path.join(corpus_dir, f"{file}.txt"), 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            temp = line.split('\t')
            group = temp[1]
            if group.find('test') != -1:
               test_id_list.append(i)
            elif group.find('train') != -1:
               train_id_list.append(i)
            label =  temp[2]
            if label in label2target:
                y.append(label2target[label])
            else:
                label2target[label] = len(label2target)
                y.append(label2target[label])

    train_size = len(train_id_list)
    test_size = len(test_id_list)

    label_list = sorted(label2target.keys(), key=lambda label: label2target[label])

    print(len(y))
    # print(label2target)
    # print(label_list)

    train_id_set = set(train_id_list)
    test_id_set = set(test_id_list)

    with open(os.path.join(corpus_dir, f"{file}.clean.txt"), 'r') as f:
        for i, doc in enumerate(f):
            if i in train_id_set:
                train_doc_list.append(doc)
                y_train.append(y[i])
            else:
                test_doc_list.append(doc)
                y_test.append(y[i])

    train_doc_list, y_train, = shuffle_doc_y_together(train_doc_list, y_train)
    test_doc_list, y_test = shuffle_doc_y_together(test_doc_list, y_test)

    doc_list = train_doc_list + test_doc_list
    y = y_train + y_test


    print(len(doc_list))

    meta['train size'] = len(y_train)
    meta['test size'] = len(y_test)
    meta['total'] = len(y_test) + len(y_train)


    with open(os.path.join(data_dir, "meta"), 'wb') as f:
        pickle.dump(meta, f)

    with open(os.path.join(data_dir, "label2target"), 'wb') as f:
        pickle.dump(label2target, f)

    with open(os.path.join(data_dir, "label_list"), 'wb') as f:
        pickle.dump(label_list, f)

    with open(os.path.join(data_dir, "y"), 'wb') as f:
        pickle.dump(y, f)

    with open(os.path.join(data_dir, "doc_list"), 'wb') as f:
        pickle.dump(doc_list, f)

    print('done')


if __name__ == "__main__":
    args = parseArgument()
    parent = os.path.dirname(__file__)
    file = args['file']
    data_dir = os.path.join(parent, "data", file)
    corpus_dir = os.path.join(parent, "data", "corpus")
    check_valid_filename(file)

    prepare(corpus_dir, data_dir, file)
    # if file == "R8":
    #     prepare_r8(data_dir)
    # elif file == "ohsumed":
    #     prepare_training_set(data_dir)

    # prepare_ohsumed("data/ohsumed")
