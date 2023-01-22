import numpy as np


class LDA:
    # Linear discriminative analysis
    def __init__(self):
        self.priorProbs = None
        self.mu = None
        self.cov = None
        self.classes = None

    def fit(self, X, Y):
        self.cov = 0
        self.mu = dict()
        self.priorProbs = dict()
        self.classes = np.unique(Y)
        for c in self.classes:
            X_c = X[Y == c]
            self.priorProbs[c] = X_c.shape[0] / X.shape[0]
            self.mu[c] = np.mean(X_c, axis=0)
            self.cov += self.priorProbs[c]*np.cov(X_c, rowvar=False)

    def predict(self, X):
        pred = list()
        length = X.shape[1]
        for x in X:
            postProbs = list()
            for c in self.classes:
                likelihood = 1 / ((2 * np.pi ** (length / 2)) * np.linalg.det(self.cov) ** 0.5) * \
                       pow(np.e, -0.5 * (x - self.mu[c]).T @ np.linalg.pinv(self.cov) @ (x - self.mu[c]))
                postProbs.append(likelihood * self.priorProbs[c])
            x_class = self.classes[np.argmax(postProbs)]
            pred.append(x_class)
        return pred
