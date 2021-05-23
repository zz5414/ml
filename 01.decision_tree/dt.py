import numpy as np
from sklearn import datasets
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt


class DecisionTreeClassifier:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def fit(self, X, y):
        pass

    def predict(self, X):
        from random import randrange
        return np.asarray([randrange(1) for input in X])


def generate_dataset():
    X, y = datasets.make_blobs(n_samples=1000,
                               centers=2,
                               n_features=2,
                               random_state=1,
                               cluster_std=3)

    for class_value in range(2):
        row_ix = np.where(y == class_value)
        plt.scatter(X[row_ix, 0], X[row_ix, 1])

    # plt.show()

    return X, y


def create_decision_tree(X, y, max_depth):
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X, y)
    return clf


if __name__ == "__main__":
    X, y = generate_dataset()
    clf = create_decision_tree(X, y, 2)
    print(clf.predict([[-10, -5]]))
    plot_decision_regions(X, y, clf=clf)
    plt.show()
