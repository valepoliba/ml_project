import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

np.random.seed(42)


class LogisticRegression:

    def __init__(self, learning_rate=1e-2, n_steps=2000, n_features=1, lmd=1):
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.theta = np.random.rand(n_features)
        self.lmd = 1
        self.lmd_ = np.full((n_features,), lmd)
        self.lmd_[0] = 0

    @staticmethod
    def _sigmoid(x):
        # return 1 / (1 + np.exp(-x))
        return expit(x)

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
        X_preds = np.c_[np.ones(X.shape[0]), X]
        return self._sigmoid(np.dot(X_preds, self.theta)) >= thrs

    @staticmethod
    def confusion_matrix(y, y_predict):
        cm = confusion_matrix(y, y_predict)
        return cm

    @staticmethod
    def roc_curvenn(y, y_predict):
        fpr, tpr, thresh = roc_curve(y, y_predict)
        auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' % auc)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess')
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid()
        plt.legend()
        plt.show()

    @staticmethod
    def accuracy(y, y_predict):
        acc = float(sum(y == y_predict) / len(y) * 100)
        return acc
