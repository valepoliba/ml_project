import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve

from neural_network import NeuralNetwork

np.random.seed(42)

df = pd.read_csv('dataset/dataset.csv')

# shuffle to avoid group bias
index = df.index
heart_df = df.iloc[np.random.choice(index, len(index))]

X = df.drop(['y'], axis=1).values
y_label = df['y'].values

train_index = round(len(X) * 0.8)

X_train = X[:train_index]
y_label_train = y_label[:train_index]

X_test = X[train_index:]
y_label_test = y_label[train_index:]

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean)/std
X_test = (X_test - mean)/std
# nn = NeuralNetwork(learning_rate=0.2, lmd=1, epochs=1000, layers=[X_train.shape[1], 100, 100, 1])
nn = NeuralNetwork(learning_rate=0.2, lmd=1, epochs=5000, layers=[X_train.shape[1], 100, 100, 1])

nn.fit(X_train, y_label_train)

nn.plot_loss()

y_predict = nn.predict(X_test)

print('Accuracy: ', nn.accuracy(y_label_test, y_predict[-1]))
print('Confusion matrix: \n', nn.confusion_matrix(y_label_test, y_predict[-1]))

# ROC
fpr, tpr, thresh = roc_curve(y_label_test, y_predict[-1])
auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' % auc)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid()
plt.legend()
plt.show()
