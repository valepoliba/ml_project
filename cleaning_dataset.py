import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def change_ds():
    try:
        input_file = open('dataset/bank-additional-full.csv', 'r')

        """items in a tuple into a string"""
        input_file = ''.join([i for i in input_file])

        input_file = input_file.replace('"', '')
        input_file = input_file.replace(';', ',')
        input_file = input_file.replace('emp.var.rate', 'emp_var_rate')
        input_file = input_file.replace('cons.price.idx', 'cons_price_idx')
        input_file = input_file.replace('cons.conf.idx', 'cons_conf_idx')
        input_file = input_file.replace('nr.employed', 'nr_employed')

        output_file = open('dataset/bank-full.csv', 'w')
        output_file.writelines(input_file)
        output_file.close()
    except RuntimeError:
        print('An error is occurred while the dataset was processing')


def check_ds():
    ds = pd.read_csv('dataset/bank-full.csv')
    print('Are there any missing value?')
    print(ds.isnull().values.any())
    print('\n')
    print('Null value for each column\n')
    print(ds.isnull().sum())
    print('\n')


print('Start dataset cleaning')

edit = input('Do you want to edit the original dataset? (y/n): \n')

if edit == 'y':
    change_ds()
    print('Replacing done\n\n')
elif edit == 'n':
    print('Dataset did not change')
else:
    print('Input error')

print('Start data check\n')

ds_file_new = Path('dataset/bank-full.csv')
try:
    my_file = ds_file_new.resolve(strict=True)
except FileNotFoundError:
    print('Dataset not found')
else:
    check_ds()

print('Conversion')

"""CONVERT CATEGORICAL TO NUMERICAL"""
df = pd.read_csv('dataset/bank-full.csv')
le = LabelEncoder()
list_to_convert = ['job', 'marital', 'default', 'housing', 'loan', 'poutcome', 'y']
for x in list_to_convert:
    label = le.fit_transform(df[x])
    df.drop(x, axis=1, inplace=True)
    df[x] = label
    print(df[x])
    print('\n')

df = df.drop(['education', 'contact', 'month', 'day_of_week', 'duration', 'pdays', 'euribor3m', 'nr_employed'], axis=1)
df.to_csv('dataset/dataset.csv', index=False)
print('Cleaning done')
