import os
import torch
import pandas
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from torch_geometric.data import Data
from sklearn.metrics import recall_score, balanced_accuracy_score, accuracy_score, roc_auc_score, roc_curve, auc, f1_score
###
def toTorchGeoData(data_in, Y, g=None):
    x = torch.tensor(data_in, dtype=torch.float)
    y = torch.tensor(Y, dtype=torch.long)
    data_out = (Data(x=x, y=y, g=g))
    return data_out


def changeDataType(lists, Y):
    data = []
    for i in range(len(lists)):
        data.append(toTorchGeoData([lists[i]], Y[i]))
    return data


def split_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_clinical(lists, excel_clinical_data):
    clinical_array = np.zeros((len(lists),10))
    column1 = "LDH"
    column2 = "pfs_2years"
    column3 = "age"
    column4 = "Ann_Arbor_stage"
    column5 = "n_extranodal_site"
    column6 = "aaIPI"
    column7 = "ECOG_scale"
    column8 = "LDH_categorical"
    column9 = "patients_id"
    column10 = "chemoterapy_regimen"
    column11 = "autologous_cell_transplant"
    column12 = "salvage_therapy"

    int_list = list(map(int, lists))
    df = pandas.read_csv(excel_clinical_data, usecols=[column1, column2, column3, column4, column5, column6, column7, column8, column10, \
                                                       column11,column12,column9], encoding='unicode_escape')
    for i in range(len(int_list)):
        select_rows_df = df.loc[df[column9] == int_list[i]]
        clinical_array[i] = select_rows_df[select_rows_df.columns[1:11]].values.tolist()[0] #Returns a list of features for each clinical_array[i] (so for each patient)

    return clinical_array


def pad_col(input, val=0, where='end'): #for survival metrics computation
    """Addes a column of `val` at the start of end of `input`."""
    if len(input.shape) != 2:
        raise ValueError(f"Only works for `phi` tensor that is 2-D.")
    pad = torch.zeros_like(input[:, :1])
    if val != 0:
        pad = pad + val
    if where == 'end':
        return torch.cat([input, pad], dim=1)
    elif where == 'start':
        return torch.cat([pad, input], dim=1)
    raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")



def get_graph(graphs, dmax):
    num_features = 3
    graph_array = np.zeros((len(graphs), num_features))
    for i in range(len(graphs)):
        g = graphs[i]
        edge_attr = g.edge_attr.tolist()
        edge_attr = [0.0 if x == 1.0 else x for x in edge_attr]
        std_edge = np.std(edge_attr)
        max_edge = max(edge_attr)
        graph_array[i] = [g.num_nodes, g.num_edges, dmax[i]]

    return graph_array


def get_labels(loc):
    column1 = ["patients_id"]
    column2 = ["pfs_2years"]
    column3 = ["age"]
    column4 = ["Ann_Arbor_stage"]
    column5 = ["n_extranodal_site"]
    column6 = ["aaIPI"]
    column7 = ["ECOG_scale"]
    column8 = ["LDH"]
    column9 = ["LDH_categorical"]

    df1 = pandas.read_csv(loc, usecols=column1,encoding= 'unicode_escape')
    col_list1 = df1['patients_id'].tolist()
    df2 = pandas.read_csv(loc, usecols=column2,encoding= 'unicode_escape')
    col_list2 = df2['pfs_2years'].tolist()
    df3 = pandas.read_csv(loc, usecols=column3, encoding='unicode_escape')
    col_list3 = df3['age'].tolist()
    df4 = pandas.read_csv(loc, usecols=column4, encoding='unicode_escape')
    col_list4 = df4['Ann_Arbor_stage'].tolist()
    df5 = pandas.read_csv(loc, usecols=column5, encoding='unicode_escape')
    col_list5 = df5['n_extranodal_site'].tolist()
    df6 = pandas.read_csv(loc, usecols=column6, encoding='unicode_escape')
    col_list6 = df6['aaIPI'].tolist()
    df7 = pandas.read_csv(loc, usecols=column7, encoding='unicode_escape')
    col_list7 = df7['ECOG_scale'].tolist()
    df8 = pandas.read_csv(loc, usecols=column8, encoding='unicode_escape')
    col_list8 = df8['LDH'].tolist()
    df9 = pandas.read_csv(loc, usecols=column9, encoding='unicode_escape')
    col_list9 = df9['LDH_categorical'].tolist()

    Y = np.array(col_list2)
    clinical = list(zip(col_list3,col_list4,col_list5,col_list6,col_list7,col_list8,col_list9))
    return  Y, np.array(col_list1), clinical


def plot_ROCAUC(n_folds, probabilities, labels, roc_kfold):
    tprs = []
    base_fpr = np.linspace(0, 1, 101)

    plt.figure(figsize=(5, 5))
    plt.axes().set_aspect('equal', 'datalim')

    for i in range(n_folds):
        y_score = probabilities[i]
        y_test = labels[i]
        fpr, tpr, _ = roc_curve(y_test, y_score)

        plt.plot(fpr, tpr, 'b', alpha=0.15, label=f'AUC {i + 1} = {roc_kfold[i]:.2f}')
        plt.legend(loc='lower right')
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    plt.plot(base_fpr, mean_tprs, 'b', label=f' Mean AUC = {np.mean(np.array(roc_kfold)):.2f}')
    plt.legend(loc='lower right')
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3,
                     label=f' Std = {np.std(np.array(roc_kfold)):.2f}')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def plot_graphic(save_path, plt, fold_count, base_fpr):
    best_lab_val = np.load(save_path + '/lab_val.npy')
    best_pred_val = np.load(save_path + '/pred_val.npy')
    best_prob_val = np.load(save_path + '/prob_val.npy')

    print('Classification')
    print('Micro F1 score ', f1_score(y_true=best_lab_val, y_pred=best_pred_val, average='micro'))
    print('Macro F1 score ', f1_score(y_true=best_lab_val, y_pred=best_pred_val, average='macro'))
    print('Weighted F1 score ', f1_score(y_true=best_lab_val, y_pred=best_pred_val, average='weighted'))

    fpr, tpr, threshold = roc_curve(best_lab_val, best_prob_val)
    best_auc_val = auc(fpr, tpr)

    plt.plot(fpr, tpr, 'b', alpha=0.15, label=f"AUC {fold_count} = %0.2f" % best_auc_val)
    plt.legend(loc='lower right')
    tpr = np.interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0

    return best_prob_val,tpr



def select_best_epoch(metrics_test, name_metrics):
    
    m = 1 # metric to optimize
    
    metrics_test_m = np.mean(metrics_test[:, :, m], axis=0) #Mean over all the loops of the AUCs (m=1) of each epoch
    best_epoch = np.argmax(metrics_test_m) #Finds the epoch where the AUC is the highest
    metrics_best_epoch = np.mean(metrics_test[:, best_epoch, :], axis=0).round(2)
    
    return best_epoch, metrics_best_epoch


def plot_training_curves(metrics_train, metrics_test, name_metrics, best_epoch):
    
    n_loop, n_epochs, n_metrics = np.shape(metrics_train)
    cst = 1.15 # constant for the confidence interval - (1.96 = 95% ), 1.15 = 75% to check 
    
    pos = range(n_epochs) 
    ticks = range(0, n_epochs, floor(n_epochs/10)) 

    for m in range(n_metrics):
    
        metrics_train_m = np.mean(metrics_train[:, :, m], axis=0).round(2)
        metrics_train_std = np.std(metrics_train[:, :, m], axis=0).round(2)
        metrics_test_m = np.mean(metrics_test[:, :, m], axis=0).round(2)
        metrics_test_std = np.std(metrics_test[:, :, m], axis=0).round(2)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(pos,metrics_train_m, '-', color='darkred', label=f"train") # ({metrics_train_m[-1]} +/- {metrics_train_std[-1]})
        ax.fill_between(pos, metrics_train_m - cst*metrics_train_std, metrics_train_m + cst*metrics_train_std, color='darkred', alpha=.1)
        ax.plot(pos, metrics_test_m, '-', color='royalblue', label=f"test ") #({metrics_test_m[-1]} +/- {metrics_train_std[-1]})
        ax.fill_between(pos, metrics_test_m - cst*metrics_test_std, metrics_test_m + cst*metrics_test_std, color='royalblue', alpha=.1)
        

        val = metrics_test_m[best_epoch]
        ax.plot([best_epoch, best_epoch], [0, val], ':', color='royalblue')
        ax.plot([0, best_epoch], [val, val], ':', color='royalblue', label=f"best epoch: {metrics_test_m[best_epoch]} +/- {metrics_test_std[best_epoch]}") 
        if m!=0:
            ax.set(ylim=[-0.05, 1.05])  #lim if not loss
        
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks)
        plt.title(name_metrics[m] + " train and test")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()


# Previous functions modified for 10 loops:
def select_best_epoch2(metrics_test, name_metrics):    
    m = 1 # metric to optimize    
    metrics_test_m = np.mean(metrics_test[:, :, :, :, m], axis=(0,1,2)) #Mean over all the loops of the AUCs (m=1) of each epoch
    best_epoch = np.argmax(metrics_test_m) #Finds the epoch where the AUC is the highest
    metrics_best_epoch = (np.mean(metrics_test[:, :, :, best_epoch, :], axis=(0,1,2))).round(2)
    
    return best_epoch, metrics_best_epoch


def plot_training_curves2(metrics_train, metrics_test, name_metrics, best_epoch, writer, save):
    
    n_test_loops, n_loop, n_loops_gsearch, n_epochs, n_metrics = np.shape(metrics_train)
    cst = 1.15 # constant for the confidence interval - (1.96 = 95% ), 1.15 = 75% to check 
    
    pos = range(n_epochs) 
    ticks = range(0, n_epochs, floor(n_epochs/10)) 

    for m in range(n_metrics):
    
        metrics_train_m = np.mean(metrics_train[:, :, :, :, m], axis=(0,1,2)).round(2)
        metrics_train_std = np.std(np.mean(metrics_train[:, :, :, :, m], axis=2), axis=(0,1)).round(2)
        metrics_test_m = np.mean(metrics_test[:, :, :, :, m], axis=(0,1,2)).round(2)
        metrics_test_std = np.std(np.mean(metrics_test[:, :, :, :, m], axis=2), axis=(0,1)).round(2)
        
        fig = plt.figure() 
        ax = fig.add_subplot(111)
        ax.plot(pos,metrics_train_m, '-', color='darkred', label=f"train") # ({metrics_train_m[-1]} +/- {metrics_train_std[-1]})
        ax.fill_between(pos, metrics_train_m - cst*metrics_train_std, metrics_train_m + cst*metrics_train_std, color='darkred', alpha=.1)
        ax.plot(pos, metrics_test_m, '-', color='royalblue', label=f"validation ") #({metrics_test_m[-1]} +/- {metrics_train_std[-1]})
        ax.fill_between(pos, metrics_test_m - cst*metrics_test_std, metrics_test_m + cst*metrics_test_std, color='royalblue', alpha=.1)
        
        val = metrics_test_m[best_epoch]
        ax.plot([best_epoch, best_epoch], [0, val], ':', color='royalblue')
        ax.plot([0, best_epoch], [val, val], ':', color='royalblue', label=f"best epoch: {metrics_test_m[best_epoch]} +/- {metrics_test_std[best_epoch]}")  
        if m!=0:
            ax.set(ylim=[-0.05, 1.05])  #lim if not loss
        
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks)
        plt.title(name_metrics[m] + " train and validation")
        plt.xlabel("Epochs")
        plt.legend()
        if(save): writer.add_figure(f"_plot/{name_metrics[m]}_train_validation", fig, close=False if(m==5) else True)
        plt.show()
