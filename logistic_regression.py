import matplotlib.pyplot as plt
import numpy as np
from numpy import mean
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


np.random.seed(42)


def confusion_matrixdef(y, y_predict):
    cm = confusion_matrix(y, y_predict)
    return cm


def roc_curvedt(y_test, X_test, model):
    model_roc_auc = roc_auc_score(y_test, model.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % model_roc_auc)
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


def logisticapplication(df):

    X = df.drop(['y'], axis=1).values
    y = df['y'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    cv = KFold(n_splits=10, random_state=1, shuffle=True)

    logreg = LogisticRegression(max_iter=1000)

    scores = cross_val_score(logreg, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    accuracy_score = mean(scores) * 100
    print('Accuracy of K-fold validation: ', accuracy_score)

    logreg.fit(X_train, y_train)

    y_predict = logreg.predict(X_test)

    print('Accuracy: ', accuracy(y_test, y_predict))
    print('Confusion matrix: \n', confusion_matrixdef(y_test, y_predict))
    roc_curvedt(y_test, X_test, logreg)
