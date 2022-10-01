import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import label_binarize

from logistic_regression import LogisticRegression
from neural_network import NeuralNetwork
from sklearn.metrics import confusion_matrix, auc, roc_curve, accuracy_score
from sklearn.metrics import roc_auc_score



np.random.seed(42)

df = pd.read_csv('dataset/dataset.csv')
"""
count_y = df['y'].value_counts()
print(count_y)
sns.countplot(x='y', data=df, palette='hls')
plt.show()
count_no_sub = len(df[df['y'] == 0])
count_sub = len(df[df['y'] == 1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("Percentage of no subscription is ", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("Percentage of subscription ", pct_of_sub*100)
"""


plt.style.use(['ggplot'])

# shuffle to avoid group bias
df = df.sample(frac=1).reset_index(drop=True)

x = df.drop(['y'], axis=1).values
y = df['y'].values

# training set and validation set
train_index = round(len(x) * 0.8)
validation_index = round(train_index * 0.7)

x_train = x[:train_index]
y_train = y[:train_index]

x_test = x[train_index:]
y_test = y[train_index:]

# zeta score normalization by column
mean = x_train.mean()
# standard deviation
std = x_train.std()

x_train = (x_train - mean)/std
x_test = (x_test - mean)/std

x_valid = x_train[validation_index:]
y_valid = y_train[validation_index:]

x_train = x_train[:validation_index]
y_train = y_train[:validation_index]

# add bias colum to the validation and train set (thetas 0 and 1)
# columns of 1 injected
x_train = np.c_[np.ones(x_train.shape[0]), x_train]
x_valid = np.c_[np.ones(x_valid.shape[0]), x_valid]


#crossval = KFold(n_splits=10, random_state=1, shuffle=True)



# basso ln, alto num di step, lambda (reach di minimum of our gradient decent)
#logistic = LogisticRegression(n_features=x_train.shape[1], learning_rate=0.0001, n_steps=15000, lmd=8)
logistic = LogisticRegression(learning_rate=0.00002, n_steps=20000, n_features=x_train.shape[1])
cost_history, cost_history_val = logistic.fit(x_train, y_train, x_valid, y_valid)

#scores = cross_val_score(logistic, x_train, y_train, scoring='accuracy', cv=10, n_jobs=1)
#print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

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
#plt.show()

preds = logistic.predict(x_test, thrs=0.5)

print(f'''Performance: {((preds == y_test).mean())}''')

# The result is telling us that we have first row sum of correct predictions and second row sum of incorrect predictions.
confusion_matrix = confusion_matrix(y_test, preds)
print('confusion_matrix')
print(confusion_matrix)

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

logit_roc_auc = roc_auc_score(y_test, logistic.predict(x_test, thrs=0.5))
fpr, tpr, thresholds = roc_curve(y_test, logistic.predict_proba(x_test, y_valid)[:, 1])
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
