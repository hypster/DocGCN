from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_20newsgroups
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy as np
import seaborn as sns
colors = ['#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700']


def plot_points(x,y):
    z = TSNE(n_components=2).fit_transform(x)
    plt.figure(figsize=(8, 8))
    sns.scatterplot(z[:,0], z[:,1], hue=y)
    # for i in range(n_categories):
    #     plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
    plt.axis('off')
    plt.show()


# categories = ['alt.atheism', 'sci.space']
categories = None
n_categories = 20 if categories is None else len(categories)
news_train = fetch_20newsgroups(subset='train', categories=categories)
news_test = fetch_20newsgroups(subset='test', categories=categories)
vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, stop_words="english")
x_train = vectorizer.fit_transform(news_train.data)
y_train = news_train.target
x_test = vectorizer.transform(news_test.data)
y_test = news_test.target

x = sp.vstack([x_train, x_test])
y = np.hstack([y_train, y_test])
plot_points(x,y)

lr = LogisticRegression(solver='lbfgs')
lr.fit(x_train, y_train)
acc = lr.score(x_test, y_test)
print("acc: %.4f" % acc)
plot_points(x_test, y_test)
