import argparse
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool
from GCN import GCN
from dataLoader import *
from prepare_ind_graph import MyOwnDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# GCN to predict graph property
class GCN_Graph(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GCN_Graph, self).__init__()

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
        out = self.linear(out)

        return out


def train(model, device, data_loader, optimizer, loss_fn):
    model.train()
    loss = 0
    for step, batch in enumerate(data_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out.to(torch.float), batch.y.view(-1))
        loss.backward()
        optimizer.step()

    return loss.item()


@torch.no_grad()
def eval(model, device, loader):
    model.eval()
    total = corr = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch)
        corr += sum(pred.argmax(dim=-1) == batch.y.view(-1)).item()
        total += len(pred)

    return corr / total


if __name__ == "__main__":

    print('Device: {}'.format(device))
    parent = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    args = parser.parse_args()
    args = vars(args)
    file = args['file']
    if file not in ('20ng', 'ned_company'):
        print("file is not recognized")
        exit()

    data_dir = os.path.join(parent, "data", file)
    model_dir = os.path.join(parent, "model_trained", file)

    dataset = MyOwnDataset(root="data/20ng")  # load the dataset
    vocab_size = len(load_word_list("20ng"))
    dataset.vocab_size = vocab_size

    train_val_size = get_20ng_train_size()  # the 20news group dataset by default
    train_val = dataset[:train_val_size]
    train_val.shuffle()
    train_val_ratio = 0.8
    train_size = int(train_val_size * train_val_ratio)
    data_train = train_val[:train_size]
    data_val = train_val[train_size:]
    data_test = dataset[train_val_size:]

    y = load_labels("20ng")
    n = len(y)
    num_tasks = max(y) + 1

    train_loader = DataLoader(data_train, batch_size=32, shuffle=True, num_workers=0)
    valid_loader = DataLoader(data_val, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(data_test, batch_size=32, shuffle=False, num_workers=0)

    args = {
        'input_dim': vocab_size,
        'device': device,
        'num_layers': 5,
        'hidden_dim': 256,
        'dropout': 0.5,
        'lr': 0.001,
        'epochs': 30,
    }

    model = GCN_Graph(args['input_dim'],
                      args['hidden_dim'],
                      num_tasks, args['num_layers'],
                      args['dropout']).to(device)

    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    loss_fn = torch.nn.CrossEntropyLoss()

    best_model = None
    best_valid_acc = 0

    for epoch in range(1, 1 + args["epochs"]):
        loss = train(model, device, train_loader, optimizer, loss_fn)
        train_acc = eval(model, device, train_loader)
        valid_acc = eval(model, device, valid_loader)
        test_acc = eval(model, device, test_loader)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            with open("model.param", "wb") as f:
                pickle.dump(model, f)

        print(f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_acc:.2f}%, '
              f'Valid: {100 * valid_acc:.2f}% '
              f'Test: {100 * test_acc:.2f}%')
