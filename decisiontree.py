import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

import pandas as pd
df = pd.read_csv('dataset/dataset.csv')
np.random.seed(42)


def confusion_matrixdef(y, y_predict):
    cm = confusion_matrix(y, y_predict)
    return cm


def roc_curvedt(y, y_predict):
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


def accuracy(y, y_predict):
    acc = float(sum(y == y_predict) / len(y) * 100)
    return acc


def decisiontreeapplication():

    X = df.drop(['y'], axis=1).values
    y = df['y'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)

    clf = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=3, max_depth=5, random_state=42)

    clf = clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)

    plt.figure(figsize=(12, 8))
    plot_tree(clf.fit(X_train, y_train))
    plt.show()

    print('Accuracy: ', accuracy(y_test, y_predict))
    print('Confusion matrix: \n', confusion_matrixdef(y_test, y_predict))
    roc_curvedt(y_test, y_predict)


decisiontreeapplication()
