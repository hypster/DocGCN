import argparse
import logging
from time import time

import torch
import torch.nn.functional as F

# The PyG built-in GCNConv
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch.nn import BatchNorm1d, LogSoftmax, ModuleList
from torch.nn.functional import relu, dropout
import copy
import random
from dataLoader import *
from scipy.sparse import identity
# only set if your env is gpu
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = True
from helper import check_valid_filename
from sklearn.metrics import f1_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):

        super(GCN, self).__init__()

        # A list of GCNConv layers
        self.convs = ModuleList(
            [GCNConv(input_dim, hidden_dim)] + [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 2)] + [
                GCNConv(hidden_dim, output_dim)])

        # A list of 1D batch normalization layers
        self.bns = ModuleList([BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)])

        # The log softmax layer
        self.softmax = LogSoftmax(1)

        # Probability of an element to be zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_weight=None):
        for i in range(len(self.convs)):
            if i == len(self.convs) - 1:
                x = self.convs[i](x, adj_t, edge_weight)
                if not self.return_embeds:
                    x = self.softmax(x)
            else:
                x = self.convs[i](x, adj_t, edge_weight)
                x = self.bns[i](x)
                x = relu(x)
                x = dropout(x, p=self.dropout, training=self.training)
        return x


@torch.no_grad()
def accuracy(pred, labels):
    """
    :param pred: tensor, last dimension C
    :param label: torch.LongTensor), target label
    :return: the accuracy of the prediction
    """
    corr = sum(pred.argmax(dim=-1) == labels)
    return corr.item() / labels.shape[0]


def train(model, data, train_idx, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    y_hat = model(data.x, data.edge_index, data.edge_weight)
    loss = loss_fn(y_hat[data.train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()
    l = loss.item()
    return l


@torch.no_grad()
def test(model, data, train_idx, val_idx, test_idx):
    model.eval()  # calling eval to pause training behavior
    pred = model(data.x, data.edge_index, data.edge_weight)
    # pred = F.log_softmax(out, 1)
    train_acc = accuracy(pred[data.train_idx], data.y[train_idx])
    valid_acc = accuracy(pred[data.val_idx], data.y[val_idx])
    test_acc = accuracy(pred[data.test_idx], data.y[test_idx])
    yhat = np.argmax(pred[data.test_idx].detach().cpu().numpy(), axis=-1)
    y = data.y[test_idx].detach().cpu().numpy()
    f_score = f1_score(y, yhat, average='micro')
    return train_acc, valid_acc, test_acc, f_score


def learn(model, data, train_idx, val_idx, test_idx, optimizer, loss_fn, epochs = 1000):
    best_valid_acc = 0
    start = time()
    for epoch in range(1, 1 + epochs):
        loss = train(model, data, train_idx, optimizer, loss_fn)
        result = test(model, data, train_idx, val_idx, test_idx)
        train_acc, valid_acc, test_acc, f_score = result
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), model.name + ".pt")
            with open(os.path.join(model_dir, model.name+".pt"), 'wb') as f:
                torch.save(model.state_dict(), f)
        print(f'Epoch: {epoch:02d}, '
                     f'Loss: {loss:.4f}, '
                     f'Train: {100 * train_acc:.2f}%, '
                     f'Valid: {100 * valid_acc:.2f}%, '
                     f'Test: {100 * test_acc:.2f}%, '
                     f'F1(Micro): {f_score:.2f}')
    total = time() - start
    print(f"total training time: {total}, "
          f"average time per epoch: {total / epochs:.2f}")

def parseArgument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()
    args = vars(args)
    return args


def generate_train_val_test_idx(start, end, file, train_val_ratio=0.9, train_test_split_ratio=0.8):
    """ load the train, validation and test index from the heterogenous adj matrix
    :param file: the dataset
    :param start: start position of the document node in the matrix
    :param end: end position of the document node in the matrix
    :param train_val_ratio: split ratio between train and validation
    :return: (train index, val index, test index)
    """
    document_idx = [i for i in range(start,
                                     end)]  # range of document nodes in the matrix, note: though train and test has been permutated each, test set follows the train set in the matrix

    if file == "ned_company":  # for company dataset there is no train test split, we first permute the document index
        document_idx = random.sample(document_idx,
                                     len(document_idx))
        train_val_size = int(len(document_idx) * train_test_split_ratio)
        train_val_idx = document_idx[:train_val_size]
    else:
        train_val_size = get_train_size(file) # get the natural train test split size
        train_val_idx = document_idx[:train_val_size]
        train_val_idx = random.sample(train_val_idx, len(train_val_idx)) # to make sure the train val set is indeed shuffled, we shuffle again the train val index


    train_size = int(len(train_val_idx) * train_val_ratio)
    train_idx = train_val_idx[: train_size]
    val_idx = train_val_idx[train_size:]
    test_idx = document_idx[train_val_size:]
    return train_idx, val_idx, test_idx


def generate_one_hot_feature(n):
    """ generate torch one hot sparse matrix
    :param n: the size of the matrix
    :return: torch one hot sparse matrix
    """
    x = identity(n, format='coo')
    i = np.vstack([x.row, x.col])
    v = x.data
    x = torch.sparse_coo_tensor(i, v, x.shape, dtype=torch.float)
    return x

if __name__ == "__main__":
    print('Device: {}'.format(device))
    parent = os.path.dirname(__file__)
    args = parseArgument()
    file = args['file']

    check_valid_filename(file)
    model_dir = os.path.join(parent, "model_trained", file)

    edge_index, edge_weight = load_edge_index_weight(file)
    y = load_labels(file)
    y = torch.LongTensor(y)
    num_class = 1 + y.max().item()
    print("num of classes: %d" % num_class)

    vocab_size = len(load_word_list(file))
    n = len(y) + vocab_size
    print("adjacency matrix size: (%d, %d)" % (n, n))

    train_idx, val_idx, test_idx = generate_train_val_test_idx(vocab_size, n, file) # l

    x = generate_one_hot_feature(n)

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, train_idx=train_idx, val_idx=val_idx,
                test_idx=test_idx, y=y,
                num_classes=num_class).to(device)

    # decrease by vocab size to get the correct index from label vector y
    train_idx = [i - vocab_size for i in train_idx]
    val_idx = [i - vocab_size for i in val_idx]
    test_idx = [i - vocab_size for i in test_idx]

    model = GCN(data.num_features, args['hidden_dim'],
                data.num_classes, args['num_layers'],
                args['dropout'], return_embeds=True).to(device)

    model_name = f"gcn_h{args['hidden_dim']}_l{args['num_layers']}_d{args['dropout']}"
    model.name = model_name
    logging.basicConfig(filename=model_name, filemode='w', level=logging.INFO, format='%(message)s')

    model.reset_parameters()
    # optimizer = torch.optim.SparseAdam(model.parameters(), lr=args['lr'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])

    loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = F.nll_loss()

    learn(model, data, train_idx, val_idx, test_idx, optimizer, loss_fn, epochs=args['epochs'])


    # learn(model_dir, file)
