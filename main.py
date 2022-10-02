import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

print('1')
edit = input('Do you want to edit the original dataset? (y/n): \n')
