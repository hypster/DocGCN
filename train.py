from model import GCN
from utils import load_data
from mxnet import nd
from mxnet import autograd
from mxnet.gluon import loss as gloss
from mxnet import gluon
from mxnet import init

def predict(net, input):
    y_hat = net(input)
    p = nd.argmax(y_hat, axis=1)
    return p

def train(net, x_train, y_train, y_test, train_mask, test_mask, loss, trainer, epochs):
    batch_size = x_train.shape[0]
    train_size = train_mask.sum().asscalar()
    test_size = test_mask.sum().asscalar()
    for i in range(epochs):
        with autograd.record():
            y_hat = net(x_train)
            l = loss(y_hat,y_train.reshape(y_hat.shape))
            # l = l * train_mask / train_mask.sum()
        l.backward()
        trainer.step(train_size)
        if (i+1) % 10 == 0:
            train_loss = l.sum().asscalar() / train_size
            test_loss = loss(y_hat, y_test).sum().asscalar()/ test_size
            pred = predict(net, x_train)
            test_acc = nd.sum((pred == nd.argmax(y_test, axis=1)) * test_mask).asscalar()
            test_acc /= test_size
            print(f"epoch {i+1}, train loss {train_loss}, test loss {test_loss}, test acc {test_acc}")

if __name__ == "__main__":
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('citeseer')
    adj = nd.array(adj.toarray())
    features = nd.array(features.toarray())
    y_train = nd.array(y_train)
    y_val = nd.array(y_val)
    y_test = nd.array(y_test)
    train_mask = nd.array(train_mask)
    val_mask = nd.array(val_mask)
    test_mask = nd.array(test_mask)


    # print(adj.shape, type(adj))

    adj = adj + nd.eye(adj.shape[0])
    D = nd.sum(adj, axis=1, keepdims=True)
    D_inv_sqrt = 1/nd.sqrt(D)
    A = D_inv_sqrt * adj * D_inv_sqrt

    in_units = features.shape[1]
    hidden_units = 16
    out_units = y_train.shape[1]
    gcn = GCN(A, in_units, hidden_units, out_units)
    # gcn.initialize()
    gcn.initialize(init=init.Xavier(), force_reinit=True)
    # print(gcn.collect_params())
    # gcn(features)
    loss = gloss.SoftmaxCrossEntropyLoss(sparse_label=False)
    epochs = 1000
    lr = 0.03
    trainer = gluon.Trainer(gcn.collect_params(), 'sgd', {'learning_rate': lr})
    train(gcn, features, y_train, y_test, train_mask, test_mask, loss, trainer, epochs)
    file_name = "gcn.params"
    gcn.save_parameters(file_name)












