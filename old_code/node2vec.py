import os.path as osp
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data
from dataLoader import *
import pickle
from random import sample
import copy
import seaborn as sns

@torch.no_grad()
def plot_points(model, y):
    model.eval()
    # z = best_model(torch.arange(data.num_nodes, device=device))
    z = model(torch.arange(len(y)))
    z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
    plt.figure(figsize=(8, 8))
    sns.scatterplot(z[:, 0], z[:, 1], s=20, hue=y)
    plt.axis('off')
    plt.show()



def main():
    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        acc = model.test(z[data.train_mask], data.y[data.train_mask],
                         z[data.test_mask], data.y[data.test_mask],
                         max_iter=150)
        return acc

    dirname = osp.join(osp.dirname(__file__), '20ng', "../data")
    path = osp.join(dirname, "edge_index")
    with open(path, 'rb') as f:
        edge_index = pickle.load(f)

    if not isinstance(edge_index, torch.LongTensor):
        edge_index = torch.LongTensor(edge_index)

    with open(osp.join(dirname, "20ng_y"), 'rb') as f:
        y = pickle.load(f)

    colors = ['#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700']
    n = len(y)
    y = torch.LongTensor(y)
    input_dimension = 20
    x = torch.arange(n)
    data = Data(x=x, edge_index=edge_index)
    data.y = y
    indexes = [i for i in range(n)]
    train_test_ratio = 0.2
    train_size = int(len(indexes) * train_test_ratio)
    mask = sample(indexes, len(indexes))
    train_mask = mask[:train_size]
    test_mask = mask[train_size:]
    train_mask = torch.LongTensor(train_mask)
    test_mask = torch.LongTensor(test_mask)
    data.train_mask = train_mask
    data.test_mask = test_mask
    data.num_classes = 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20,
                     context_size=5, walks_per_node=10,
                     num_negative_samples=1, p=1, q=0.5, sparse=True).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    best_valid_acc = 0
    best_model = model

    # plot_points(colors)

    for epoch in range(1, 301):
        loss = train()
        acc = test()
        if acc > best_valid_acc:
            best_valid_acc = acc
            best_model = copy.deepcopy(model)
            with open(osp.join(osp.dirname(__file__), '../model_trained', "20ng_node2vec_emb_300.param"), 'wb') as f:
                pickle.dump(best_model, f)

        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')


if __name__ == "__main__":
    # main()

    p = osp.join(osp.dirname(__file__), "../model_trained", "20ng_node2vec.param")
    with open(p, 'rb') as f:
        model = pickle.load(f)
    p = osp.join(osp.dirname(__file__), "20ng", "../data", "20ng_y")
    with open(p, 'rb') as f:
        y = pickle.load(f)

    plot_points(model, y)
