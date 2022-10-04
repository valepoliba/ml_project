import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logistic_regression as la
import neural_network as nna
import decisiontree as dta

np.random.seed(42)
plt.style.use(['ggplot'])
df = pd.read_csv('dataset/dataset.csv')

count_y = df['y'].value_counts()
print('Count of target: \n', count_y)
sns.countplot(x='y', data=df, palette='hls')
plt.show()
count_no_sub = len(df[df['y'] == 0])
count_sub = len(df[df['y'] == 1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print('Percentage of no subscription is ', pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print('Percentage of subscription ', pct_of_sub*100)

print('1-Logistic regression | 2-Neural Network | 3-Decision Tree')
edit = input('Input: \n')

if edit == '1':
    la.logisticapplication(df)
elif edit == '2':
    nna.nnapplication(df)
elif edit == '3':
    dta.decisiontreeapplication(df)
else:
    print('Error!')
