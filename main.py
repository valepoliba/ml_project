import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


np.random.seed(42)

df = pd.read_csv('dataset/bank-full.csv')
"""print(df.describe())
print(df)"""

"""CONVERT CATEGORICAL TO NUMERICAL"""
le = LabelEncoder()
"""education, contact', 'month', 'day_of_week',"""
list_to_convert = ['job', 'marital', 'default', 'housing', 'loan', 'poutcome', 'y']
for x in list_to_convert:
    label = le.fit_transform(df[x])
    df.drop(x, axis=1, inplace=True)
    df[x] = label
    print(df[x])
    print('\n')

df.to_csv('dataset/test.csv')
