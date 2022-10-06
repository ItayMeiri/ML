import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# calculates the Lp norm in a vectorized form. 'a' here is the whole training sample in KNN
def lp_distance(a, b, order=2):
    return np.linalg.norm(a - b, ord=order, axis=1)


# Checks for a majority in an array of labels. In our algorithm, this would be the the labels fo the K nearest neighbors
def vote(labels):
    if np.sum(labels) >= 0:
        return 1
    return -1


# Loads the two_circle data
def load_data(path):
    df = pd.read_csv(path, sep=" ", names=["d1", "d2", "label"])
    X, y = np.array(df[df.columns[0:2]]), np.array(df[df.columns[2]])
    return X, y


# KNN class classifier. Instead of just doing the algorithm itself.
class KNN:
    def __init__(self, k=1, p=2):
        self.X = None
        self.y = None
        self.k = k
        self.p = p

    def fit(self, X, y):
        self.X = X
        self.y = y

    # Actual KNN algorithm!
    def predict(self, data):
        predictions = np.zeros(data.shape[0])
        for i, sample in enumerate(data):
            # calculate the distances of every point in train to the sample we want to categorize
            distances = lp_distance(self.X, sample, self.p)
            # find the minimum K values index
            indices = np.argpartition(distances, self.k)[:self.k]
            predictions[i] = vote(self.y[indices])
        return predictions

    def score(self, data, labels):
        predictions = self.predict(data)
        return np.count_nonzero(labels == predictions) / labels.size


# keeping a running average of everything
k_running_avg = {1: [0, 0], 3: [0, 0], 5: [0, 0], 7: [0, 0], 9: [0, 0]}
p_running_avg = {1: [0, 0], 2: [0, 0], np.inf: [0, 0]}

X, y = load_data("two_circle.txt")

for i in range(100):
    print("=======")
    print("Iteration number:", i)
    # We have 150 samples, so this would split them randomly in each iteration to 75-75
    # we're using random_state=i in order to get a different seed every iteration, but for multiple runs to have the
    # same output.
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.50, random_state=i)
    clf = KNN()
    clf.fit(Xtrain, ytrain)
    for k in [1, 3, 5, 7, 9]:
        print("*******")
        print("K=" + str(k), "neighbors")
        clf.k = k
        for p in [1, 2, np.inf]:
            clf.p = p
            test_acc = clf.score(Xtest, ytest)
            dist_acc = clf.score(X, y)

            print("L_" + str(p), "Test accuracy:", test_acc, "Overall distribution accuracy:", dist_acc)
            print("Correctly identified:", np.sum(clf.predict(Xtest) == ytest))
            k_running_avg[k][0] += test_acc
            k_running_avg[k][1] += dist_acc
            p_running_avg[p][0] += test_acc
            p_running_avg[p][1] += dist_acc

    print("*******")
    print("=======")

for k in [1, 3, 5, 7, 9]:
    print("for k =", k, "empirical error:", 1 - k_running_avg[k][0] / 300, "true error:",
          1- k_running_avg[k][1] / 300)

for p in [1, 2, np.inf]:
    print("for p =", p, "empirical error:", 1 - p_running_avg[p][0] / 500, "true error:",
          1 - p_running_avg[p][1] / 500)
