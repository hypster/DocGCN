from mxnet import nd
from mxnet.gluon import nn

class GCN_layer(nn.Block):
    def __init__(self, graph_matrix, in_units, units, **kwargs):
        super(GCN_layer, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        # self.bias = self.params.get('bias', shape=(1,units))
        self.A = graph_matrix

    def forward(self, x):
        return nd.dot(self.A, nd.dot(x, self.weight.data()))


def GCN(A, in_units, hidden_units, out_units):
    net = nn.Sequential()
    net.add(GCN_layer(A,in_units, hidden_units))
    net.add(nn.Activation('relu'))
    net.add(GCN_layer(A, hidden_units, out_units))
    return net
