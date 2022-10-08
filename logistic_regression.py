import numpy as np
import evaluation as ev
from datetime import datetime
from numpy import mean
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score

np.random.seed(42)


def logisticapplication(df):

    start_time = datetime.now()

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

    print('Accuracy: ', ev.accuracy(y_test, y_predict))
    ev.confusion_matrixdef(y_test, y_predict)
    ev.roc_curvedt(y_test, X_test, logreg)

    end_time = datetime.now()

    print('Execution time (second): {}'.format((end_time - start_time)))
