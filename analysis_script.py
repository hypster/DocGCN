import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
reg = re.compile('[\d]*[.][\d]+')
def extract_log(file):
    loss = []
    train_acc = []
    val_acc = []
    test_acc = []
    f1_score = []
    arr = []
    with open(file, "r") as f:
        lines = f.readlines()
        for l in lines[:-1]:
            temp = []
            for i, obj in enumerate(reg.finditer(l)):
                temp.append(float(obj[0]))

            arr.append(temp)

    loss, train_acc, val_acc, test_acc, f1_score = list(zip(*arr))


    return {'loss': loss, 'train acc': train_acc, 'val acc': val_acc, 'test acc': test_acc, 'f1 score': f1_score}

def plot(performance):
    loss = performance['loss']
    if len(loss):
        plt.plot(loss)
        plt.show()

if __name__ == "__main__":
    log_map = {}
    parent = "train_log/gcnp/"
    for f in os.listdir(parent):
        print(f)
        log_map[f] = extract_log(os.path.join(parent, f))

    print(log_map)
    # plot(performance)
