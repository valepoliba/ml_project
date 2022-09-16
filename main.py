import pandas as pd
import numpy as np


np.random.seed(42)

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

# data score normalization by column
mean = x_train.mean()
std = x_train.std()

x_train = (x_train - mean)/std
x_test = (x_test - mean)/std

x_valid = x_train[validation_index:]
y_valid = y_train[validation_index:]

x_train = x_train[:validation_index]
y_train = y_train[:validation_index]

# bias colum to the validation and train features
x_train = np.c_[np.ones(x_train.shape[0]), x_train]
x_valid = np.c_[np.ones(x_valid.shape[0]), x_valid]