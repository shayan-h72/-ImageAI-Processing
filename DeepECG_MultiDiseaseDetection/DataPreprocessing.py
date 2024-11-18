import numpy as np
import matplotlib.pyplot as plt
import os
import wfdb 
import pandas as pd

file_directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(file_directory,'mit-bih-database')

def process_data():

    data_files = []
    exclude_dirs = ['mitdbdir', 'x_mitdb']

    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.endswith('.dat'):
                data_files.append(os.path.splitext(os.path.join(root, file))[0])
    
    return data_files

def read_data(data_files):
    df = pd.DataFrame()
    for i in range(len(data_files)):
        #record = wfdb.rdrecord(data_files[i])
        annotation = wfdb.rdann(data_files[i], 'atr')

        sym = annotation.symbol
        values, counts = np.unique(sym, return_counts=True)
        tem_df = pd.DataFrame({'sym':values, 'val':counts})
        df = pd.concat([df,tem_df], axis=0)

    return annotation, df

data_files = process_data()
annotations, df = read_data(data_files)

print('some information about dataset...')
print(df.groupby('sym').val.sum().sort_values(ascending=False))