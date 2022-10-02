import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression

"""
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import label_binarize

from logistic_regression import LogisticRegression
from neural_network import NeuralNetwork
from sklearn.metrics import confusion_matrix, auc, roc_curve, accuracy_score
from sklearn.metrics import roc_auc_score
"""


np.random.seed(42)
plt.style.use(['ggplot'])

df = pd.read_csv('dataset/dataset.csv')

df = df.sample(frac=1).reset_index(drop=True)

x = df.drop(['y'], axis=1).values
y = df['y'].values

train_index = round(len(x) * 0.8)
validation_index = round(train_index * 0.7)

x_train = x[:train_index]
y_train = y[:train_index]

x_test = x[train_index:]
y_test = y[train_index:]

mean = x_train.mean()
std = x_train.std()

x_train = (x_train - mean)/std
x_test = (x_test - mean)/std

x_valid = x_train[validation_index:]
y_valid = y_train[validation_index:]

x_train = x_train[:validation_index]
y_train = y_train[:validation_index]

x_train = np.c_[np.ones(x_train.shape[0]), x_train]
x_valid = np.c_[np.ones(x_valid.shape[0]), x_valid]

#logistic = LogisticRegression(n_features=x_train.shape[1], learning_rate=0.0001, n_steps=15000, lmd=8)
logistic = LogisticRegression(learning_rate=0.00002, n_steps=20000, n_features=x_train.shape[1])
cost_history, cost_history_val = logistic.fit(x_train, y_train, x_valid, y_valid)

print(f'''Thetas: {*logistic.theta,}''')
print(f'''Final train BCE {cost_history[-1]:.3f}''')
print(f'''Final validation BCE {cost_history_val[-1]:.3f}''')

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_ylabel('J(Theta)')
ax.set_xlabel('Iterations')
c, = ax.plot(range(logistic.n_steps), cost_history, 'b.')
cv, = ax.plot(range(logistic.n_steps), cost_history_val, 'r+')
c.set_label('Train cost')
cv.set_label('Valid cost')
ax.legend()
plt.show()

y_predict = logistic.predict(x_test, thrs=0.5)

print('Accuracy: ', logistic.accuracy(y_test, y_predict))
print('Confusion matrix: \n', logistic.confusion_matrix(y_test, y_predict))


"""


x = df.drop(['y'], axis=1).values
y = df['y'].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))



# The result is telling us that we have first row sum of correct predictions and second row sum of incorrect predictions.
confusion_matrix = confusion_matrix(y_test, y_pred)
print('confusion_matrix')
print(confusion_matrix)


#ROC
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

"""




"""
#ROC
logit_roc_auc = roc_auc_score(y_test, preds)
fpr, tpr, thresholds = roc_curve(y_test, preds)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
"""
"""
fpr = dict()
tpr = dict()
roc_auc = dict()

fpr, tpr, _ = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

# Compute micro-average ROC curve and ROC area
#fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), preds.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()
"""
"""
fpr, tpr, thresh = roc_curve(y_test, preds)
auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' % auc)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid()
plt.legend()
#plt.show()
"""