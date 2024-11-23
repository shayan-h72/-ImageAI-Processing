import numpy as np
import matplotlib.pyplot as plt
import os
import wfdb 
import pandas as pd
import random

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
        record = wfdb.rdrecord(data_files[i])
        annotation = wfdb.rdann(data_files[i], 'atr')

        sym = annotation.symbol
        values, counts = np.unique(sym, return_counts=True)
        tem_df = pd.DataFrame({'record':record,'annotation':annotation,'sym':values, 'val':counts})
        df = pd.concat([df,tem_df], axis=0)

    return df

def plot_sample(df, show_ab = False):

    sample = df.sample(n=1)
    sample_rec = sample.iloc[0]['record']
    wfdb.plot_wfdb(record=sample_rec, title='sample ECG signal')
    signal = sample_rec.p_signal
    
    if show_ab:
        plt.figure(figsize=(10,4))
        abnormal = ['L','R','V','/','A','f','F','j','a','E','J','e','S']

        sym = sample.iloc[0]['sym']
        an_sample = sample.iloc[0]['annotation'].sample
        ab_index = [b for a,b in zip(sym,an_sample) if a in abnormal][:10]
        singnal_range = np.arange(len(signal))

        if len(ab_index) > 0:
            window = 1000
            ceneter = ab_index[0]
            left, right =  max(0, ceneter - window)  , min(len(signal), ceneter + window)

            if len(signal[left:right]) > 0:
                plt.plot(singnal_range[left:right], signal[left:right,0], label='ECG', linestyle='-', color='blue')

                normal_indices = np.intersect1d(np.arange(left, right), an_sample)
                plt.scatter(singnal_range[normal_indices], signal[normal_indices,0], color='green', label='Normal', marker='o')

                abnormal_indices = np.intersect1d(np.arange(left, right), ab_index)
                plt.scatter(singnal_range[abnormal_indices], signal[abnormal_indices,0],color='red', label='Abnormal', marker='o')

                plt.xlim(left, right)
                plt.ylim(signal[left:right, 0].min() - 0.05, signal[left:right, 0].max() + 0.05)


                plt.xlabel('Time Index', fontsize=12)
                plt.ylabel('ECG Signal', fontsize=12)
                plt.title('ECG Signal with Annotations', fontsize=14)
                plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
                plt.grid(True, linestyle='--', alpha=0.6)

                plt.tight_layout()
                plt.show()
            
            else:
                print('No data in the selected range.')
    
        else:
            print('No abnormal annotations found.')

def get_data(plot = False, show_ab=True):
    data_files = process_data()
    df = read_data(data_files)

    print('some information about dataset...')
    print(df.groupby('sym').val.sum().sort_values(ascending=False))

    if plot:
        plot_sample(df, show_ab=show_ab)
    
    return df
