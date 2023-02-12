import numpy as np


class NaiveBayes:
    # Naive Bayes
    def __init__(self):
        self.priorProbs = None
        self.mu = None
        self.std = None
        self.classes = None

    def fit(self, X, Y):
        self.std = dict()
        self.mu = dict()
        self.priorProbs = dict()
        self.classes = np.unique(Y)
        for c in self.classes:
            X_c = X[Y == c]
            self.priorProbs[c] = X_c.shape[0] / X.shape[0]
            self.mu[c] = np.mean(X_c, axis=0)
            self.std[c] = np.sqrt(np.sum(np.square(X_c - self.mu[c]), axis=0) / X_c.shape[0])

    def predict(self, X):
        pred = list()
        for x in X:
            log_posts = list()
            for c in self.classes:
                # likelihood = (1/(np.sqrt(2*np.pi)*self.std[c])) * np.exp((-(x-self.mu[c])**2)/(2*self.std[c]**2))
                likelihood = (1/(np.sqrt(2*np.pi)*self.std[c])) * np.exp((-(x-self.mu[c])**2)/(2*self.std[c]**2))
                log_posts.append(np.sum(np.log(likelihood)) + np.log(self.priorProbs[c]))
            x_class = self.classes[np.argmax(log_posts)]
            pred.append(x_class)
        return pred


