import argparse
from random import random
from time import time

from sklearn.metrics import f1_score
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, global_sort_pool
from GCN import GCN
from dataLoader import *
from helper import check_valid_filename
from buildDocumentGraph import MyOwnDataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# GCN to predict graph property
class GCNP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GCNP, self).__init__()

        # Node embedding model, the input_dim and output_dim are set to hidden_dim
        self.gnn_node = GCN(input_dim, hidden_dim,
                            hidden_dim, num_layers, dropout, return_embeds=True)

        self.pool = global_mean_pool  # Initialize the self.pool to global mean pooling layer

        self.linear = torch.nn.Linear(hidden_dim, output_dim)  # Output layer
        self.input_dim = input_dim

    def reset_parameters(self):
        self.gnn_node.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, batched_data):
        x, edge_index, edge_weight, batch = batched_data.x, batched_data.edge_index, batched_data.edge_weight, batched_data.batch
        # x stores the identity of the node, needs to transform to one-hot for training
        rows = []
        cols = []
        v = []
        for row, col in enumerate(x):
            rows.append(row)
            cols.append(col.item())
            v.append(1)

        i = np.vstack([rows, cols])
        n = len(x)
        x = torch.sparse_coo_tensor(i, v, size=(n, self.input_dim), dtype=torch.float).to(device)

        out = self.gnn_node(x, edge_index, edge_weight)
        out = self.pool(out, batch)
        # # new dropout layer after pool
        # out = dropout(out, p=0.5, training=self.training)
        out = self.linear(out)

        return out


def train(model, data_loader, optimizer, loss_fn):
    model.train()
    loss = 0
    for step, batch in enumerate(data_loader):
        # print(batch.batch.max()+1, batch.y.shape)
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out.to(torch.float), batch.y.view(-1))
        loss.backward()
        optimizer.step()

    return loss.item()


@torch.no_grad()
def eval(model, loader, compute_f1 = False):
    model.eval()
    total = corr = 0
    yhat = np.array([])
    y = np.array([])
    f_score = None
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch)
        yhat_batch = pred.argmax(dim=-1)
        corr += torch.sum(yhat_batch == batch.y.view(-1)).item()
        total += len(pred)
        if compute_f1:
            yhat = np.concatenate([yhat, yhat_batch.detach().cpu().numpy().reshape(-1)])
            y = np.concatenate([y, batch.y.detach().cpu().numpy().reshape(-1)])

    if compute_f1:
        f_score = f1_score(y, yhat, average='micro')
    return corr / total, f_score

@torch.no_grad()
def eval_f1(model, loader):
    model.eval()


def parseArgument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr',type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--train_val_ratio',type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--pool', default='mean')
    args = parser.parse_args()
    args = vars(args)
    return args

def learn(model, optimizer, loss_fn, train_loader, valid_loader, test_loader, epochs = 30):
    best_valid_acc = 0
    start = time()
    for epoch in range(1, 1 + epochs):
        loss = train(model, train_loader, optimizer, loss_fn)
        train_acc, _ = eval(model, train_loader)
        valid_acc, _ = eval(model, valid_loader)
        test_acc, f_score = eval(model, test_loader, compute_f1 = True)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            with open(os.path.join(model_dir, model.name + ".pt"), 'wb') as f:
                torch.save(model.state_dict(), f)

        print(f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_acc:.2f}%, '
              f'Valid: {100 * valid_acc:.2f}%, '
              f'Test: {100 * test_acc:.2f}%, '
              f'F1(Micro): {f_score:.4f}')
    total = time() - start
    print(f"total training time: {total}, "
          f"average time per epoch: {total / epochs:.2f}")


def load_train_val_test_data(dataset, file, train_val_ratio = 0.9):
    if file == "ned_company":
        train_val_size = int(0.8 * len(
            dataset))  # for custom dataset where there is no fixed train test split, use 0.8 of total data size for train val set
    else:
        train_val_size = get_train_size(file) # other dataset comes with a default split

    train_val = dataset[:train_val_size]
    train_val.shuffle()  # shuffle the train validation set first
    train_size = int(train_val_size * train_val_ratio)
    data_train = train_val[:train_size]
    # only for the experiment
    # for data in data_train:
        # if random() < 1:
        #     data.y = torch.LongTensor([int(random() * 20)])

    data_val = train_val[train_size:]
    data_test = dataset[train_val_size:]
    return data_train, data_val, data_test


if __name__ == "__main__":

    print('Device: {}'.format(device))
    parent = os.path.dirname(__file__)
    args = parseArgument()

    file = args['file']

    check_valid_filename(file)

    data_dir = os.path.join(parent, "data", file)
    print(data_dir)
    model_dir = os.path.join(parent, "model_trained", file)

    dataset = MyOwnDataset(root=data_dir)  # load the dataset

    vocab_size = len(load_word_list(file))
    dataset.vocab_size = vocab_size
    data_train, data_val, data_test = load_train_val_test_data(dataset, file, args['train_val_ratio'])

    y = load_y(file)
    n = len(y)
    num_class = max(y) + 1

    batch_size = args['batch_size']
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=0)

    model = GCNP(vocab_size,
                 args['hidden_dim'],
                 num_class, args['num_layers'],
                 args['dropout']).to(device)

    model_name = f"gcnp_{file}_h{args['hidden_dim']}_l{args['num_layers']}_d{args['dropout']}_l{args['lr']}_b{args['batch_size']}"
    model.name = model_name
    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])

    loss_fn = torch.nn.CrossEntropyLoss()

    learn(model, optimizer, loss_fn, train_loader, valid_loader, test_loader, epochs=args["epochs"])


