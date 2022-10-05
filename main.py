import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logistic_regression as la
import neural_network as nna
import decisiontree as dta
"""
plt.style.use(['ggplot'])
df = pd.read_csv('dataset/bank-full.csv')

count_y = df['y'].value_counts()
print('Target count: \n', count_y)
sns.countplot(x='y', data=df, palette='hls')
plt.title('Target count')
plt.show()
count_no_sub = len(df[df['y'] == 'no'])
count_sub = len(df[df['y'] == 'yes'])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print('Percentage of no subscription is ', pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print('Percentage of subscription ', pct_of_sub*100)

pd.crosstab(df.job, df.y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.show()

table = pd.crosstab(df.marital, df.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.show()

df.age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
"""
df_modified = pd.read_csv('dataset/dataset.csv')

#print('1-Logistic regression | 2-Neural Network | 3-Decision Tree')
#edit = input('Input: \n')
edit = '2'
if edit == '1':
    la.logisticapplication(df_modified)
elif edit == '2':
    nna.nnapplication(df_modified)
elif edit == '3':
    dta.decisiontreeapplication(df_modified)
else:
    print('Error!')
