import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def data_formatter(dataframe):
    data_list=[]
    for i in range(len(dataframe)):
        tmp_list=''
        for k,v in dataframe.iloc[i].items():
            if int(v) == 1:
                if tmp_list == '':
                    tmp_list = tmp_list + k
                else:
                    tmp_list = tmp_list + ',' + k
        if tmp_list:
            data_list.append(tmp_list)
    return data_list

def co_occurrence_counter(data_list):
    co_data_dict = {}
    for co_data in data_list:
        labels = co_data.split(',')
        
        for i,label in enumerate(labels):
            for label_ in labels[i:]:
                A, B = label, label_
                if A > B:
                    A, B = B, A  
                co_label = A+','+B
                if co_label not in list(co_data_dict.keys()):
                    co_data_dict[co_label] = 1
                else:
                    co_data_dict[co_label] += 1
                
    return co_data_dict

def generate_matrix(co_data_dict, labels):
    matrix = pd.DataFrame(np.zeros((len(labels),len(labels))), columns=labels, index=labels)
    for key, value in co_data_dict.items():
        A = key.split(',')[0]
        B = key.split(',')[1]
        matrix.loc[A, B] = value
        matrix.loc[B, A] = value
    return matrix.astype(int)
    
def plot_co_matrix(df, dropDuplicates = True):
    if dropDuplicates:    
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

    sns.set_style(style = 'white')
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    if dropDuplicates:
        sns.heatmap(df, mask=mask, cmap=cmap, 
                square=True,
                annot=True,
                fmt='g',
                linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
    else:
        sns.heatmap(df, cmap=cmap, 
                square=True,
                annot=True,
                fmt='g',
                linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)

