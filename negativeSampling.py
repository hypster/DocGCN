from torch import nn
import torch
def simpleSampling(G, num_neg_samples):
    """returns a list of negative edges.
    :param G: Networkx Graph
    :param num_neg_samples: The number of sampled negative edges
    :return: negative edge list, shape (2, number of edges)
    """
    # neg_edge_list = []
    neg_edge_set = set()
    n = G.number_of_nodes()
    for i in range(n):
        for j in range(n):
            if i == j or (i, j) in G.edges or (j, i) in G.edges or (i, j) in neg_edge_set or (j, i) in neg_edge_set:
                continue
            neg_edge_set.add((i, j))
            if len(neg_edge_set) == num_neg_samples:
                return list(neg_edge_set)


torch.manual_seed(1)
def create_node_emb(num_node, embedding_dim=16):
    """create the node embedding matrix.
    The weight matrix is initialized under uniform distribution.
    :param num_node:
    :param embedding_dim:
    :return: torch.nn.Embedding layer
    """
    emb = nn.Embedding(num_embeddings=num_node, embedding_dim = embedding_dim)
    emb.weight.data = torch.rand(num_node, embedding_dim)
    return emb