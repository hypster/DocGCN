import os
import pickle

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

from model import GCN
from mxnet import nd
from mxnet import autograd
from mxnet.gluon import loss as gloss
from mxnet import gluon
from mxnet import init
import numpy as np
from scipy import sparse


def predict(net, input):
    """use the model for prediction

    :param net: the model
    :param input:
    :return: the prediction
    """
    y_hat = net(input)
    p = nd.argmax(y_hat, axis=1)
    return p


def train(net, features, y_all, train_mask, test_mask, loss, trainer, epochs):
    """Train the model

    :param net: model
    :param features: node features
    :param y_train: label
    :param y_test: label
    :param train_mask: the masking array that tells which are from train data
    :param test_mask: similarly masking array for testing data
    :param loss: the loss function
    :param trainer: the optimization function
    :param epochs:
    :return: the trained model
    """
    train_size = train_mask.sum().asscalar()
    test_size = test_mask.sum().asscalar()
    for i in range(epochs):
        with autograd.record():
            y_hat = net(features)
            l = loss(y_hat, y_all, train_mask)
            # l = l * train_mask / train_mask.sum()
        l.backward()
        trainer.step(train_size)  # divide by the train size to update the parameter
        if (i + 1) % 10 == 0:
            train_loss = l.sum().asscalar() / train_size
            test_loss = loss(y_hat, y_all, test_mask).sum().asscalar() / test_size  # similar for test data
            pred = predict(net, features)
            correct = (pred == y_all.astype('float32'))
            train_acc = nd.sum(correct * train_mask).asscalar() / train_size
            test_acc = nd.sum(correct * test_mask).asscalar() / test_size

            print(f"epoch {i + 1}, train loss {train_loss}, test loss {test_loss},"
                  f"train acc {train_acc}, test acc {test_acc}")

if __file__:
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
else:
    parent = os.getcwd()
    data_dir = os.path.join(parent, '20ng', 'data')
with open(os.path.join(data_dir, '20ng_y'), 'rb') as f:
    Y = pickle.load(f)

with open(os.path.join(data_dir,'20ng_word_list'), 'rb') as f:
    word_list = pickle.load(f)

vocab_size = len(word_list)

n_cls = max(Y) + 1

y_train, y_test = train_test_split(Y, train_size=0.5, shuffle=False)


with open(os.path.join(data_dir, '20ng_adj'), 'rb') as f:
    adj = pickle.load(f)

is_doc = np.zeros(shape=(adj.shape[0],))
start = adj.shape[0] - len(Y)
for i in range(len(Y)):
    is_doc[start + i] = 1

train_mask = np.zeros(shape=(adj.shape[0], ))
y_train_len = len(y_train)
for i in range(y_train_len):
    train_mask[i + vocab_size] = 1

train_mask = nd.array(train_mask)

test_mask = np.zeros(shape=(adj.shape[0], ))
y_test_len = len(y_test)
for i in range(y_test_len):
    test_mask[i + y_train_len + vocab_size] = 1

test_mask = nd.array(test_mask)

y_all = np.zeros(shape=(adj.shape[0],))
for i in range(len(Y)):
    y_all[vocab_size+i] = Y[i]

y_all = nd.array(y_all)


# y_train = np.eye(n_cls)[y_train]
# y_test = np.eye(n_cls)[y_test]


# y_train = nd.zeros(shape=(adj.shape[0], 20))
#
# y_test = nd.zeros(shape=(adj.shape[0], 20))
# y_train_size = len(train_label)
# train_mask = nd.zeros(shape=(adj.shape[0],))
# test_mask = nd.zeros(shape=(adj.shape[0],))

# for i in range(len(train_label)):
#     label = train_label[i]
#     y_train[start + i, label] = 1
#     train_mask[start + i] = 1
#
# for i in range(len(test_label)):
#     label = test_label[i]
#     y_test[start + y_train_size + i, label] = 1
#     test_mask[start + y_train_size + i] = 1

# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('citeseer')

"""compute the normalized adjacency matrix"""
adj = adj + sparse.eye(adj.shape[0])

D = sparse.diags(adj.sum(axis=1).A1)

D1 = D.power(-0.5)

A = D1 * adj * D1

in_units = 300

features = nd.random.uniform(shape=(adj.shape[0], 300))

# features = nd.eye(adj.shape[0])  # simply identity matrix for feature matrix

# print(adj.shape, type(adj))

hidden_units = 200  # hyperparameter
out_units = n_cls

gcn = GCN(A, in_units, hidden_units, out_units)
# gcn.initialize()
# gcn.initialize(init=init.Xavier(), force_reinit=True)
gcn.initialize(init=init.Normal(0.01), force_reinit=True)
# print(gcn.collect_params())
# gcn(features)
loss = gloss.SoftmaxCrossEntropyLoss(sparse_label=True) # loss function
epochs = 100
lr = 0.1
trainer = gluon.Trainer(gcn.collect_params(), 'sgd', {'learning_rate': lr})
train(gcn, features, y_all, train_mask, test_mask, loss, trainer, epochs)
file_name = "gcn.ng20.params"
gcn.save_parameters(file_name)
