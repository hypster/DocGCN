import seaborn as sns
import matplotlib.pyplot as plt
def extract_log(file="log/graph_20ng_256_5"):
    loss = []
    train_acc = []
    val_acc = []
    test_acc = []
    with open(file, "r") as f:
        for l in f:
            i = l.index("Loss")
            if i != -1:
                i += 5
                loss.append(float(l[i:i+7]))
            i = l.index("Train")
            if i != -1:
                i += 6
                train_acc.append(float(l[i:i+6]))
            i = l.index("Valid")
            if i != -1:
                i += 6
                val_acc.append(float(l[i:i+6]))
            i = l.index("Test")
            if i != -1:
                i += 5
                test_acc.append(float(l[i:i + 6]))

    return {'loss': loss, 'train acc': train_acc, 'val acc': val_acc, 'test acc': test_acc}

def plot(performance):
    loss = performance['loss']
    if len(loss):
        sns.lineplot(loss)
        plt.show()

if __name__ == "__main__":
    performance = extract_log()
    plot(performance)
