import matplotlib.pyplot as plt
import numpy as np
import evaluation as ev
from numpy import mean
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, KFold, cross_val_score

np.random.seed(42)


def decisiontreeapplication(df):

    X = df.drop(['y'], axis=1).values
    y = df['y'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)

    cv = KFold(n_splits=10, random_state=1, shuffle=True)

    dectree = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=3, max_depth=5, random_state=42)

    scores = cross_val_score(dectree, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    accuracy_score = mean(scores) * 100
    print('Accuracy of K-fold validation: ', accuracy_score)

    dectree = dectree.fit(X_train, y_train)

    y_predict = dectree.predict(X_test)

    plt.figure(figsize=(12, 8))
    plot_tree(dectree.fit(X_train, y_train))
    plt.show()

    print('Accuracy: ', ev.accuracy(y_test, y_predict))
    ev.confusion_matrixdef(y_test, y_predict)
    ev.roc_curvedt(y_test, X_test, dectree)
