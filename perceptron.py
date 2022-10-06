import numpy as np
import pandas as pd

path = "two_circle.txt"

# loading data and extracting data and labels - X, y
df = pd.read_csv(path, sep=" ", names=["d1", "d2", "label"])
X, y = np.array(df[df.columns[0:2]]), np.array(df[df.columns[2]])

# Unused area: adding a bias, splitting to train/test
# Split requires training with Xtrain, ytrain instead of X, y and evaluating with Xtest, ytest!

# X = np.c_[X, np.ones(X.shape[0])]  # adds a column of -1
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)


def perceptron(data, labels, rounds=1):
    w = np.zeros(data.shape[1])  # initializing weights to 0

    for t in range(rounds):
        mistake = False  # Mistake flag will exit the algorithm if none is found.
        for idx, sample in enumerate(data):
            if w @ sample > 0:  # modern numpy matrix multiplication, giving the dot product.
                if labels[idx] == -1:  # guess +,  actually -
                    w = w - sample  # w_t+1 = w_t - sample
                    mistake = True
                    break
            else:
                if labels[idx] == 1:  # guess -, actually +
                    w = w + sample  # w_t+1 = w_t + sample
                    mistake = True
                    break
        if not mistake:
            print("No mistakes, exiting algorithm")
            return w
    return w


def evaluate(X, y, w):
    # 1. X @ weights.T - returns a matrix of the dot products for every sample
    # 2. np.sign maps them to -1 or 1
    # 3. == y will return a matrix, where every element is True/False based on whether the prediction from step (2)
    #   matched the true label
    # 4. np.count_nonzero will count how many 'True' items are there - those are the correct predictions!
    correct = np.count_nonzero(np.sign(X @ w.T) == y)
    wrong = y.size - correct
    return correct, wrong


# How many rounds do we need until we stop getting errors?
# This isn't an efficient implementation, but I wanted to see the results for every round without altering the
# perceptron algorithm to be verbose
converged = False
last_weights = perceptron(X, y, 0)
epoch = 1
while not converged:
    print("Number of rounds:", epoch)
    new_weights = perceptron(X, y, epoch)
    print("New weights:", new_weights)
    print("Evaluate:", evaluate(X, y, new_weights))
    if np.array_equiv(last_weights, new_weights):
        print("No updates on the", epoch, "round.")
        print("Final vector:", new_weights)
        converged = True
    epoch += 1
    last_weights = new_weights
    print("")
