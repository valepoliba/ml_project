import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay

np.random.seed(42)


def confusion_matrixdef(y, y_predict):
    plt.style.use(['default'])
    cm = confusion_matrix(y_predict, y, labels=[1, 0])
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_display.plot()
    plt.title('Confusion Matrix')
    plt.show()


def roc_curvedt(y_test, X_test, model):
    plt.style.use(['ggplot'])
    model_roc_auc = roc_auc_score(y_test, model.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='Model area (area = %0.2f)' % model_roc_auc)
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
