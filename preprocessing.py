import numpy as np
import os

def read_citation_network(filename):
    data_dir = "/"
    node_labels = {}
    with open(os.path.join(data_dir, filename, filename+'.node_labels')) as f:
        cnt = 0
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            idx, label  = line.split(',')
            idx = int(idx)
            label = int(label)
            idx -= 1 # turn into 0 index
            node_labels[idx] = label
            cnt += 1

        Y = np.zeros((cnt, ), dtype=np.int32)
        for idx, label in node_labels:
            Y[idx] = label


    adj = np.zeros((cnt, cnt), dtype=np.int32)

    with open(os.path.join(data_dir, filename, filename + '.edges')) as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            x,y,_= line.split(",")
            x = int(x)
            y = int(y)
            x -= 1
            y -= 1
            adj[x][y] = 1

    return adj, Y

if __name__ == "__main__":
    adj, Y = read_citation_network("data/citeseer")


