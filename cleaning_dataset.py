print('Start dataset cleaning')
try:
    input_file = open('dataset/bank-additional-full.csv', 'r')

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
except:
    print('An error is occurred while the dataset was processing')

print('Replacing done')
print('')
print('Start data check')


print('Cleaning done')
