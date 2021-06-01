from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
from networkx import karate_club_graph
import os



def visualize_emb(emb, G):
    """
    :param emb: torch Embedding
    :param G: networks Graph
    :return:
    """
    if isinstance(emb, torch.nn.Embedding):
        X = emb.weight.data.numpy()
    else:
        X = emb
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    plt.figure(figsize=(6, 6))
    club1_x = []
    club1_y = []
    club2_x = []
    club2_y = []
    for node in G.nodes(data=True):
        if node[1]['club'] == 'Mr. Hi':
            club1_x.append(components[node[0]][0])
            club1_y.append(components[node[0]][1])
        else:
            club2_x.append(components[node[0]][0])
            club2_y.append(components[node[0]][1])
    plt.scatter(club1_x, club1_y, color="red", label="Mr. Hi")
    plt.scatter(club2_x, club2_y, color="blue", label="Officer")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    G = karate_club_graph()
    p = os.path.join(os.path.dirname(__file__), "data/karate/data/karate.emd")
    index_arr = []
    emb_arr = []
    with open(p, 'r') as f:
        line = f.readline().strip()
        n, m = line.split(" ")
        for i, line in enumerate(f):
            l = line.strip().split(" ")
            idx = int(l[0])
            emb = [float(e) for e in l[1:]]
            index_arr.append(idx)
            emb_arr.append(emb)

    order_arr = [k for k, v in sorted(enumerate(index_arr), key=lambda x: x[1])]
    emb_arr = [emb_arr[i] for i in order_arr]
    print(len(emb_arr), len(emb_arr[0]))
    visualize_emb(emb_arr, G)
