import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from logistic_regression import LogisticRegression
from neural_network import NeuralNetwork

np.random.seed(42)
plt.style.use(['ggplot'])

df = pd.read_csv('dataset/dataset.csv')

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

def neural():
    nn = NeuralNetwork(learning_rate=0.001, epochs=1000, lmd=1, layers=[x_train.shape[1], 100, 1])

    nn.fit(x_train, y_train)

    # nn.plot_loss()

    preds = nn.predict(x_test)

    print(nn.accuracy(y_test, preds[-1]))


def logistic():
    # basso ln, alto num di step, lambda (reach di minimum of our gradient decent)
    logistic = LogisticRegression(n_features=x_train.shape[1], learning_rate=0.0001, n_steps=15000, lmd=8)
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
    plt.show()

    preds = logistic.predict(x_test, thrs=0.5)
    print(f'''Performance: {((preds == y_test).mean())}''')


logistic()
# neural()
