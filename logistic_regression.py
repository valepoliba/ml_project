import numpy as np
# inversa della log function
from scipy.special import expit

np.random.seed(42)


class LogisticRegression:

    def __init__(self, learning_rate=1e-2, n_steps=2000, n_features=1, lmd=1):
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.theta = np.random.rand(n_features)
        self.lmd = 1
        self.lmd_ = np.full((n_features,), lmd)
        self.lmd_[0] = 0

    def _sigmoid(self, x):
        #return 1 / (1 + np.exp(-x))
        return expit(x)

    # lambda to provide regularization approach
    def fit(self, X, y, X_val, y_val):
        m = len(X)
        theta_history = np.zeros((self.n_steps, self.theta.shape[0]))
        cost_history = np.zeros(self.n_steps)
        cost_history_val = np.zeros(self.n_steps)

        for step in range(0, self.n_steps):
            preds = self._sigmoid(np.dot(X, self.theta))
            preds_val = self._sigmoid(np.dot(X_val, self.theta))

            error = preds - y

            self.theta = self.theta - ((1 / m) * self.learning_rate * (np.dot(X.T, error) + (self.theta.T * self.lmd_)))
            theta_history[step, :] = self.theta.T

            loss = -(1/m) * (np.dot(y.T, np.log(preds)) + np.dot((1-y.T), np.log(1-preds)))
            loss_val = -(1 / m) * (np.dot(y_val.T, np.log(preds_val)) + np.dot((1 - y_val.T), np.log(1 - preds_val)))

            reg = (self.lmd/(2*m)) * np.dot(self.theta.T[1:], self.theta[1:])

            cost_history[step] = loss + reg
            cost_history_val[step] = loss_val + reg

        return cost_history, cost_history_val

    def predict(self, X, thrs):
        # = bias column
        X_preds = np.c_[np.ones(X.shape[0]), X]
        return self._sigmoid(np.dot(X_preds, self.theta)) >= thrs