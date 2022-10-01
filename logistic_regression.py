import numpy as np
# inversa della log function
from scipy.special import expit
from sklearn.utils.extmath import softmax, safe_sparse_dot
from sklearn.utils.validation import check_is_fitted, check_array

np.random.seed(42)


class LogisticRegression:

    def __init__(self, learning_rate=1e-2, n_steps=2000, n_features=1, lmd=1, multi_class='auto', solver='lbfgs', classes_=None):
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.theta = np.random.rand(n_features)
        self.lmd = 1
        self.lmd_ = np.full((n_features,), lmd)
        self.lmd_[0] = 0
        self.multi_class = multi_class
        self.solver = solver
        self.classes_ = classes_
        self.n_features = n_features

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

    def _predict_proba_lr(self, X):
        """Probability estimation for OvR logistic regression.

        Positive class probabilities are computed as
        1. / (1. + np.exp(-self.decision_function(X)));
        multiclass is handled by normalizing that over all classes.
        """
        prob = self.decision_function(X)
        expit(prob, out=prob)
        if prob.ndim == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob

    def decision_function(self, X):

        check_is_fitted(self)

        X = check_array(X, accept_sparse='csr')

        n_features = self.n_features - 1
        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], n_features))
        self.intercept_ = np.zeros(n_classes)
        scores = safe_sparse_dot(X, n_features,
                                 dense_output=True) + self.intercept_
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict_proba(self, X, y):
        check_is_fitted(self)
        self.classes_ = np.unique(y)
        ovr = (self.multi_class in ["ovr", "warn"] or
               (self.multi_class == 'auto' and (self.classes_.size <= 2 or
                                                self.solver == 'liblinear')))
        if ovr:
            return self._predict_proba_lr(X)
        else:
            decision = self.decision_function(X)
            if decision.ndim == 1:
                # Workaround for multi_class="multinomial" and binary outcomes
                # which requires softmax prediction with only a 1D decision.
                decision_2d = np.c_[-decision, decision]
            else:
                decision_2d = decision
            return softmax(decision_2d, copy=False)

    def fittest(self, X, y, X_val, y_val):
        m = len(X)
        m_val = len(X_val)
        theta_history = np.zeros((self.n_steps, self.theta.shape[0]))
        cost_history = np.zeros(self.n_steps)
        cost_history_val = np.zeros(self.n_steps)

        for step in range(0, self.n_steps):
            preds = self._sigmoid(np.dot(X, self.theta))
            preds_val = self._sigmoid(np.dot(X_val, self.theta))

            error = preds - y

            cost_history[step] = -(1 / m) * (np.dot(y.T, np.log(preds)) + np.dot((1 - y.T), np.log(1 - preds)))
            cost_history_val[step] = -(1 / m) * (
                        np.dot(y_val.T, np.log(preds_val)) + np.dot((1 - y_val.T), np.log(1 - preds_val)))

            self.theta = self.theta - ((1 / m) * self.learning_rate * np.dot(X.T, error))
            theta_history[step, :] = self.theta.T

        return cost_history, cost_history_val
