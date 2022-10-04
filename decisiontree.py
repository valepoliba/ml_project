import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean


np.random.seed(42)


def confusion_matrixdef(y, y_predict):
    cm = confusion_matrix(y, y_predict)
    return cm


def roc_curvedt(y_test, X_test, clf):
    clf_roc_auc = roc_auc_score(y_test, clf.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % clf_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def accuracy(y, y_predict):
    acc = float(sum(y == y_predict) / len(y) * 100)
    return acc


def decisiontreeapplication(df):

    X = df.drop(['y'], axis=1).values
    y = df['y'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)

    cv = KFold(n_splits=10, random_state=1, shuffle=True)

    clf = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=3, max_depth=5, random_state=42)

    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    score2 = mean(scores) * 100
    print('Accuracy of K-fold validation: ', score2)

    clf = clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)

    plt.figure(figsize=(12, 8))
    plot_tree(clf.fit(X_train, y_train))
    plt.show()

    print('Accuracy: ', accuracy(y_test, y_predict))
    print('Confusion matrix: \n', confusion_matrixdef(y_test, y_predict))
    roc_curvedt(y_test, X_test, clf)
