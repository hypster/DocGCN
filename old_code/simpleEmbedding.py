import torch
from torch import nn
from torch.optim import SGD
from negativeSampling import simpleSampling, create_node_emb
import networkx
from helper import visualize_emb


def accuracy(pred, labels):
    """This function takes the pred tensor (the resulting tensor after sigmoid) and the label tensor (torch.LongTensor).
    :param pred: tensor
    :param label: tensor
    :return: the accuracy of the prediction
    """
    pred = torch.sigmoid(pred)
    predicted = (pred > 0.5).int()
    correct = (predicted == labels).sum().item()
    return correct / labels.shape[0]


def train(emb, loss_fn, train_label, train_edge, epochs = 500, learning_rate = 0.1):
    """Train the embedding layer
    :param emb: embedding
    :param loss_fn: loss function
    :param train_label:
    :param train_edge:
    :return:
    """
    n = len(train_label)
    optimizer = SGD(emb.parameters(), lr=learning_rate, momentum=0.9)
    for i in range(epochs):
        # performs the dot product, pred shape len(edge_list)
        pred = torch.mul(emb(train_edge[0]), emb(train_edge[1])).sum(dim=1)
        optimizer.zero_grad()
        loss = loss_fn(pred, train_label)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            acc = accuracy(pred, train_label)
            print("loss: %.4f accuracy: %.4f" % (loss, acc))

def edge_list_to_tensor(edge_list):
    """transforms the edge_list to tensor.
    :param edge_list: The input edge_list is a list of tuples
    :return: the resulting tensor should have the shape [2 x len(edge_list)]
    """
    return torch.tensor(edge_list, dtype=torch.int64).T


if __name__ == "__main__":
    G = networkx.karate_club_graph()
    # positive edge index
    pos_edge_index = edge_list_to_tensor(list(G.edges))
    print("The neg_edge_index tensor has shape {}".format(pos_edge_index.shape))
    # print("The pos_edge_index tensor has sum value {}".format(torch.sum(pos_edge_index)))
    # here we sample equal number of negative edges as positive edges
    neg_edge_list = simpleSampling(G, pos_edge_index.shape[1])
    neg_edge_index = edge_list_to_tensor(neg_edge_list)
    print("The neg_edge_index tensor has shape {}".format(neg_edge_index.shape))

    # Generate the positive and negative labels
    pos_label = torch.ones(pos_edge_index.shape[1], )
    neg_label = torch.zeros(neg_edge_index.shape[1], )

    # Concat positive and negative labels into one tensor
    train_label = torch.cat([pos_label, neg_label], dim=0)

    # Concat positive and negative edges into one tensor
    # Since the network is very small, we do not split the edges into val/test sets
    train_edge = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.BCELoss()
    emb = create_node_emb(G.number_of_nodes(), embedding_dim=16)
    print("Embedding: {}".format(emb))

    visualize_emb(emb, G)


    train(emb, loss_fn, train_label, train_edge)
    visualize_emb(emb, G)