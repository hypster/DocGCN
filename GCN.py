import argparse
import logging
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

logging.basicConfig(filename='log_tmp', filemode='w', level=logging.INFO, format='%(message)s')


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):

        super(GCN, self).__init__()

        # A list of GCNConv layers
        self.convs = ModuleList([GCNConv(input_dim, hidden_dim)] + [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers-2)] + [GCNConv(hidden_dim, output_dim)])

        # A list of 1D batch normalization layers
        self.bns = ModuleList([BatchNorm1d(hidden_dim) for _ in range(num_layers-1)])

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

    def forward(self, x, adj_t, edge_weight = None):
        for i in range(len(self.convs)):
            if i == len(self.convs)-1:
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
    loss = loss_fn(y_hat[data.train_idx], data.y[train_idx]) # reshape(-1) to make target 1d vector
    loss.backward()
    optimizer.step()
    l = loss.item()
    loss = None
    return l


# Test function here
@torch.no_grad()
def test(model, data, train_idx, val_idx, test_idx):
    """this function tests the model by
    using the given split_idx and evaluator.
    :param model:
    :param data:
    :param train_idx: the index from the document data, not to be confused with data.train_idx which stores the index based on the adjacency matrix
    :param evaluator:
    :return:
    """
    model.eval() # calling eval to pause dropout
    pred = model(data.x, data.edge_index, data.edge_weight)
    # pred = F.log_softmax(out, 1)
    train_acc = accuracy(pred[data.train_idx], data.y[train_idx])
    valid_acc = accuracy(pred[data.val_idx], data.y[val_idx])
    test_acc = accuracy(pred[data.test_idx], data.y[test_idx])
    return train_acc, valid_acc, test_acc


def main(device, parent, file):
    edge_index, edge_weight = load_edge_index_weight(file)
    edge_index = torch.LongTensor(edge_index)
    edge_weight = torch.FloatTensor(edge_weight)
    y = load_labels(file)
    vocab_size = len(load_word_list(file))
    print("pmi sub matrix size: (%d, %d)" % (vocab_size, vocab_size))
    n = len(y) + vocab_size
    print("adjacency matrix size: (%d, %d)" % (n, n))


    document_idx = [i for i in range(vocab_size,
                                     n)]  # document index in the adjacency matrix, the start and end index of the bottom right sub matrix



    train_idx, val_idx, test_idx = load_train_val_test_idx(file, document_idx)

    # total_idx = random.sample(document_idx, len(document_idx))
    # train_val_ratio = 0.8
    # train_size = int(len(total_idx) * train_val_ratio)
    # train_idx = total_idx[: train_size]
    # val_idx = total_idx[train_size:]

    # x = torch.ones(size=(len(adj), 20))
    # x = F.one_hot(torch.arange(n))  # one-hot encoding
    x = identity(n, format='coo')
    i = np.vstack([x.row, x.col])
    v = x.data
    x = torch.sparse_coo_tensor(i, v, x.shape, dtype=torch.float)

    num_classes = max(y) + 1
    y = torch.LongTensor(y)

    print("num of classes: %d" % num_classes)
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, train_idx=train_idx, val_idx=val_idx, test_idx = test_idx, y=y,
                num_classes=num_classes).to(device)

    # del edge_index, edge_weight, y

    # Make the adjacency matrix to symmetric

    # split_idx = dataset.get_idx_split()
    # train_idx = split_idx['train'].to(device)

    train_idx = [i - vocab_size for i in train_idx]
    val_idx = [i - vocab_size for i in val_idx]
    test_idx = [i - vocab_size for i in test_idx]

    args = {
        'device': device,
        'num_layers': 2,
        'hidden_dim': 32,
        'dropout': 0.5,
        'lr': 0.01,
        'epochs': 1000,
    }

    # input_dim, hidden_dim, output_dim, num_layers,
    # dropout, return_embeds = False
    model = GCN(data.num_features, args['hidden_dim'],
                data.num_classes, args['num_layers'],
                args['dropout'], return_embeds=True).to(device)

    # reset the parameters to initial random value
    model.reset_parameters()
    # torch.optim.sparse_adam
    # optimizer = torch.optim.SparseAdam(model.parameters(), lr=args['lr'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = F.nll_loss()

    best_valid_acc = 0

    for epoch in range(1, 1 + args["epochs"]):
        loss = train(model, data, train_idx, optimizer, loss_fn)

        result = test(model, data, train_idx, val_idx, test_idx)
        train_acc, valid_acc, test_acc = result
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            with open(os.path.join(parent, file + '_' + str(args['hidden_dim']) + "_model"), 'wb') as f:
                pickle.dump(model, f)
        logging.info(f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_acc:.2f}%, '
              f'Valid: {100 * valid_acc:.2f}% '
              f'Test: {100 * test_acc:.2f}% ')




def load_train_val_test_idx(file, document_idx, train_val_ratio = 0.8):
    """ load the train, validation and test index from the heterogenous adj matrix
    :param file: the dataset
    :param document_idx: the index in the heterogenous adj matrix
    :param train_val_ratio: split ratio between train and validaton
    :return: (train index, val index, test index)
    """
    document_idx = random.sample(document_idx,
                                 len(document_idx))  # for personal dataset we first shuffle the document
    if file == "20ng":
        train_val_size = get_20ng_train_size() # the 20news group dataset by default split the train and test based on the date of the text, as this split is used in the leaderbord
    else:
        train_val_size = int(len(document_idx) * 0.9) # and simply use 90% of data for train and val

    train_val = document_idx[:train_val_size]
    train_val_idx = random.sample(train_val, len(train_val)) # then shuffle the train val set
    train_size = int(len(train_val_idx) * train_val_ratio)
    train_idx = train_val_idx[: train_size]
    val_idx = train_val_idx[train_size:]
    test_idx = document_idx[train_val_size:]
    return train_idx, val_idx, test_idx


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))
    parent = os.path.join(os.path.dirname(__file__), "model_trained")
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    args = parser.parse_args()
    args = vars(args)
    file = args['file']
    if file not in ('20ng', 'ned_company'):
        print("file is not recognized")
        exit()

    main(device, parent, file)


