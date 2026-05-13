import os
import sys
import json
import pycox
import torch
import pandas
import random
import datetime
import argparse
import numpy as np
import torch.nn as nn
import torchtuples as tt
import matplotlib.pyplot as plt
from pycox.models import DeepHitSingle, CoxTime, CoxPH
from pycox.evaluation import EvalSurv
from sklearn_pandas import DataFrameMapper
from pycox.datasets import metabric, support
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader, DataListLoader
# from lifelines.fitters.cox_time_varying_fitter import CoxTimeVaryingFitter 
from torch.utils.tensorboard import SummaryWriter
###
sys.path.insert(1, './') 
from functions.survival import analysis_bs, kaplanmeyer_curves, kaplanmeyer_curves_classif
from functions.Scheduler import LRScheduler
from functions.MilDataset import collate
from functions.graph_making import graph_loop
from functions.miscellaneous import get_labels
from functions.loops_functions import generate_model, train_val_test, variables_init, balance_data, split_data, gridsearch_params, disp_save, write, train_val_test_surv
from functions.train_val_test_splits import test_split, train_val_split
from parameters import parameters
plt.close('all')

#Code arguments (only needed if we are on a cluster):
parser = argparse.ArgumentParser("loops_main.py")
parser.add_argument("grid_search_loop", help="The current grid search loop number.", nargs='?', const=-1, default=-1)
parser.add_argument("time_job", help="The time when the whole grid search was launched (format: date +%Y_%m_%d-%Hh%M_%Ss), used to retrieve the files resulting from each grid search.", nargs='?', const='', default='')
parser.add_argument("cluster", help="Whether we run these codes on a cluster, thus separating all the grid search loops in different jobs (True) or not (False).", nargs='?', const=True, default=False)
args = parser.parse_args()

# #GPU use:
# if (torch.cuda.is_available()):  #We do not use GPU because it give results that are not always reproducible
#     dev = "cuda:0"  
# else: 
#     dev = "cpu" 
# device = torch.device(dev) 
# torch.cuda.set_device(device)


# General variables initialization:
date = datetime.datetime.today()
random.seed(42)
np.random.seed(42)            #IMPORTANT
_ = torch.manual_seed(123)
_parameters = parameters()
model_type = _parameters.model
n_test_splits = 5 if(_parameters.mean5tests) else 1  
n_time_intervals = _parameters.n_time_intervals if(_parameters.loss_fct in _parameters.survival_list) else -1
m = nn.Sigmoid()
# Y, patient_list, clinical = get_labels(_parameters.clinical_data)

if(_parameters.DB!='METABRIC' and _parameters.DB!='SUPPORT'):
    patient_list = [np.int64(_parameters.patients_to_use[i]) for i in range(len(_parameters.patients_to_use))]
    df = pandas.read_csv(_parameters.DB, usecols=["patients_id", "pfs_2years", "lesion_id", "pfs", "pfs_event"], encoding='unicode_escape')
    df = df[df["patients_id"].isin(patient_list)]
    if(not _parameters.largest): df = df[df["lesion_id"]==1]
    bin_Y = np.array(df['pfs_2years'])  
else:
    df_metabric = metabric.read_df()
    bin_Y = np.array(df_metabric[['duration','event']]) #not binary but we must keep the same stratification as the survival tests
    patient_list = np.int64(range(len(bin_Y))) 

###   ####   #####   TO CHANGE ?
if(_parameters.loss_fct in _parameters.classif_list):
    Y = bin_Y
elif(_parameters.loss_fct in _parameters.survival_list):
    if(_parameters.DB == 'METABRIC'):
        df_metabric = metabric.read_df()
        Y = np.array(df_metabric[['duration','event']])
    elif(_parameters.DB == 'SUPPORT'):
        df_support = support.read_df()
        Y = np.array(df_support[['duration','event']])
    else:
        Y = np.array(df[['pfs', 'pfs_event']])
    discretization_scheme = _parameters.discretization_scheme #If discrete-time survival analysis
    if(_parameters.loss_fct == 'DeepHitS'):
        labtrans = DeepHitSingle.label_transform(n_time_intervals, scheme=discretization_scheme) 
    elif(_parameters.loss_fct == 'CoxTime'):
        labtrans = CoxTime.label_transform() 
else:
    raise ValueError('parameters.loss_function is not in parameters.classif_list nor in parameters.survival_list (in parameters.py)')
print(Y.shape) #(561,2)
#print(np.unique(Y[:,1], return_counts=True))  #check censure


if(_parameters.saving_results and (_parameters.DB!='METABRIC' and _parameters.DB!='SUPPORT' or _parameters.model=='RSF_METABRIC')):
    log_dir = f"./save/results/{model_type}/{args.time_job}{_parameters.name_save}/tensorboard" if(args.cluster) else f"./save/results/{model_type}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/tensorboard/"
    if(args.cluster):
        try:
            os.makedirs(f'./save/results/{model_type}/{args.time_job}{_parameters.name_save}/tensorboard')
        except FileExistsError:
            pass
        try:
            os.makedirs(f'./tmp/{args.time_job}{_parameters.name_save}')
        except FileExistsError:
            pass
    else:
        try:
            os.makedirs(f'./save/results/{model_type}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/tensorboard')
        except FileExistsError:
            pass
        try:
            os.makedirs(f'./save/results/{_parameters.model}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/splits')
        except FileExistsError:
            pass
elif(_parameters.saving_results):
    log_dir = f"./save/results/{_parameters.DB}/{args.time_job}{_parameters.name_save}/tensorboard" if(args.cluster) else f"./save/results/{_parameters.DB}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/tensorboard/"
    if(args.cluster):
        try:
            os.makedirs(f'./save/results/{_parameters.DB}/{args.time_job}{_parameters.name_save}/tensorboard')
        except FileExistsError:
            pass
        try:
            os.makedirs(f'./tmp/{args.time_job}{_parameters.name_save}')
        except FileExistsError:
            pass
    else:
        try:
            os.makedirs(f'./save/results/{_parameters.DB}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/tensorboard')
        except FileExistsError:
            pass
        # TO EXPORT SPLITS
        try:
            os.makedirs(f'./save/results/{_parameters.DB}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/splits')
        except FileExistsError:
            pass

#To ensure survival parameters are shared between the MLP and the EPN (for consistence)
if(_parameters.loss_fct in _parameters.survival_list and _parameters.saving_results):
    with open(log_dir+'../survival_parameters.json', 'w') as file:
        json.dump({'loss_fct':_parameters.loss_fct, 'n_time_intervals':n_time_intervals, 'discretization_scheme': discretization_scheme, 'stratif':_parameters.stratif}, file)

if (model_type in _parameters.MIL_list): from torch.utils.data import DataLoader 
best_parameters_list = []
roc_auc_max, max_ep = 0,0

# Variables initialization (by model):
parameters_list, feature_array, bonus_array, ids, labels, classical, list_centers, radiomics, edge_weights, clinical_array, clin_not_stand = variables_init(model_type, _parameters, patient_list, Y)

# Graphs computation:
if (model_type in _parameters.GNN_list and _parameters.DB!='METABRIC' and _parameters.DB!='SUPPORT'):
    graphs = []
        # we standardize based on the whole dataset to do the computation only once
    for i in range(len(parameters_list['alpha'])):
        print(parameters_list['alpha'][i])
        G, Graphs_alpha = graph_loop(len(Y), classical, list_centers, radiomics, Y, parameters_list['alpha'][i], _parameters)#, device) #Graphs_alpha is an array of graphs
        graphs.append(Graphs_alpha)
    print('Graphs computation finished')

# Grid search preparation:
grid_param = gridsearch_params(parameters_list)
n_loops_gsearch = 1 if(args.cluster) else len(grid_param)

# Metrics initialization:
if(_parameters.loss_fct in _parameters.classif_list):
    name_metrics = ["loss", "au-roc", "accuracy", "sensitivity", "specificity", "balanced_accuracy"]
    confusion_mat = np.zeros((n_test_splits, _parameters.n_loops, len(np.unique(Y)), len(np.unique(Y))))
elif(_parameters.loss_fct in _parameters.survival_list):
    name_metrics = ["loss", "c_index", "IBS"] #IBS for Integrated Brier score: Brier score treated independently because not of the same shape as others
    name_bonus_metrics = ['cindex_uncensored', 'ibs_uncensored', 'ibs_censored']
    metrics_bonus_test_saved = np.zeros((n_test_splits, _parameters.n_loops, len(name_bonus_metrics)))
    LEN_BS = 100
    bs = np.zeros((n_test_splits, _parameters.n_loops, LEN_BS))  #bs = Brier score; used only for the test
    confusion_mat = None
else:
    raise ValueError('parameters.loss_function is not in parameters.classif_list nor in parameters.survival_list (in parameters.py)')
nb_metrics = len(name_metrics)
metrics_train = np.zeros((n_test_splits, _parameters.n_loops, n_loops_gsearch, max(parameters_list["nb_epochs"]), nb_metrics))
metrics_val = np.zeros((n_test_splits, _parameters.n_loops, n_loops_gsearch, max(parameters_list["nb_epochs"]), nb_metrics))
metrics_test = np.zeros((n_test_splits, _parameters.n_loops, nb_metrics))
p_values = np.zeros((n_test_splits, _parameters.n_loops, n_loops_gsearch, 4))
if(_parameters.loss_fct in _parameters.survival_list):time_grid_test_cont = np.zeros((n_test_splits, LEN_BS))
confusion_mat = np.zeros((n_test_splits, _parameters.n_loops, len(np.unique(Y)), len(np.unique(Y))))
if(_parameters.loss_fct in _parameters.classif_list and _parameters.DB=='METABRIC'): confusion_mat = np.zeros((n_test_splits, _parameters.n_loops, 2, 2))


for test_s in range(n_test_splits):                                                                                                                 
    # Test split:  #random seed init=4444                                                                                                             
    if(_parameters.loss_fct in _parameters.survival_list or _parameters.DB=='METABRIC'): #We want to keep the same stratification when we use METABRIc in classification
        match _parameters.stratif:
            case 'nostratif':
                stratif = None
            case 'stratif2yPFStest':
                stratif = bin_Y
            case 'stratifcens':
                stratif = Y[:,1]
            case _:
                raise ValueError("parameters.stratif is not in ['nostratif', 'stratif2yPFStest', 'stratifcens'] (in parameters.py)")
    else:
        stratif = bin_Y
    test_list, others_list, test_labs, others_labs = train_test_split(feature_array, Y, test_size=(1-_parameters.prop_test), random_state=4444*(test_s+1), stratify=stratif)\
        if(_parameters.pat_to_use!='patAsw545' and not (_parameters.pat_to_use=='patfull561' and _parameters.sameSplit561As583)) else test_split(feature_array, bin_Y, _parameters.patients_to_use)  #we want to get the same splits as Aswathi

    #Stratification for training/validation
    match _parameters.stratif:
        case 'nostratif':
            stratif = None
        case 'stratif2yPFStest':
            stratif = None
        case 'stratifcens':
            stratif = others_labs[:,1]
        case _:
            raise ValueError("parameters.stratif is not in ['nostratif', 'stratif2yPFStest', 'stratifcens'] (in parameters.py)")        
        
    for l in range(_parameters.n_loops):
        value_early_stop_max = -1

        # Train/validation split:
        if((_parameters.loss_fct in _parameters.survival_list or _parameters.DB=='METABRIC') and (_parameters.pat_to_use!='patAsw545' and not (_parameters.pat_to_use=='patfull561' and _parameters.sameSplit561As583))):
            train_list, val_list, train_labs, val_labs = train_test_split(others_list, others_labs, test_size=_parameters.prop_val/(1-_parameters.prop_test), random_state=4444*l, stratify=stratif)   #We can't stratify on others_labs if we have survival labels 
        elif(_parameters.pat_to_use!='patAsw545' and not (_parameters.pat_to_use=='patfull561' and _parameters.sameSplit561As583)):    
            train_list, val_list, train_labs, val_labs = train_test_split(others_list, others_labs, test_size=_parameters.prop_val/(1-_parameters.prop_test), random_state=4444*l, stratify=others_labs)
        else:
            train_val_split(others_list, others_labs, _parameters.patients_to_use, l)
        print(len(train_list), len(val_list), len(test_list))

        # # Standardizing the data:
        # train_data, val_data, test_data, input_model_size, Graphs_created, classical, list_centers, radiomics = standardize(model_type, train_list, val_list, test_list, train_labels, val_labels, test_labels, classical, list_centers, radiomics, bonus_array, clinical_array, ids, labels, Y, parameters_list, _parameters, edge_weights)
        train_mask_nocens_before_mediantime, val_mask_nocens_before_mediantime = np.ones_like(patient_list), np.ones_like(patient_list)
        # Discretizing the survival data if needed, or standardizing it for CoxTime:
        if(_parameters.loss_fct == 'DeepHitS' and _parameters.model != 'RSF_METABRIC'):
            y_train = labtrans.fit_transform(train_labs[:,0], train_labs[:,1])
            y_val = labtrans.transform(val_labs[:,0], val_labs[:,1])
            # y_test = labtrans.transform(test_labs[:,0], test_labs[:,1]) #Transform needed? Is it a problem otherwise? #Yes it is a problem: removed
            y_test = test_labs.transpose()[0], test_labs.transpose()[1]
            time_intervals_limits = labtrans.cuts
            train_labels, val_labels, test_labels = y_train, y_val, y_test
        elif(_parameters.loss_fct == 'CoxTime' and _parameters.model != 'RSF_METABRIC'): 
            y_train = labtrans.fit_transform(train_labs[:,0], train_labs[:,1])
            y_val = labtrans.transform(val_labs[:,0], val_labs[:,1])
            # y_test = labtrans.transform(test_labs[:,0], test_labs[:,1]) #Transform needed? Is it a problem otherwise? #Yes it is a problem: removed
            y_test = test_labs.transpose()[0], test_labs.transpose()[1]
            train_labels, val_labels, test_labels = y_train, y_val, y_test
        elif(_parameters.loss_fct == 'DeepSurv' and _parameters.model != 'RSF_METABRIC'):
            train_labels = train_labs.transpose()[0], train_labs.transpose()[1]
            val_labels = val_labs.transpose()[0], val_labs.transpose()[1]
            test_labels = test_labs.transpose()[0], test_labs.transpose()[1]
        else:
            train_labels, val_labels, test_labels = train_labs, val_labs, test_labs            
            y_test = test_labs.transpose()[0], test_labs.transpose()[1]
            if(_parameters.DB == 'METABRIC' and _parameters.model != 'RSF_METABRIC'):
                ############################Deal with converting survival values to classification
                mediantime = np.median(train_labs[:,0])
                train_labels, val_labels, test_labels = np.array([int(x) for x in train_labs[:,0]>mediantime]), np.array([int(x) for x in val_labs[:,0]>mediantime]), np.array([int(x) for x in test_labs[:,0]>mediantime])
                train_mask_nocens_before_mediantime = [(train_labels[i] or int(train_labs[i,1])) for i in range(len(train_labels))] 
                val_mask_nocens_before_mediantime = [(val_labels[i] or int(val_labs[i,1])) for i in range(len(val_labels))] 


        if(_parameters.loss_fct in _parameters.survival_list):
            time_grid_train_cont = np.linspace(np.min(train_labs[:,0]), np.max(train_labs[:,0]), LEN_BS)
            time_grid_val_cont = np.linspace(np.min(val_labs[:,0]), np.max(val_labs[:,0]), LEN_BS)
            time_grid_test_cont[test_s] = np.linspace(np.min(test_labs[:,0]), np.max(test_labs[:,0]), LEN_BS)
        # time_grid_train = np.linspace(np.min(train_labels[:,0]), np.max(train_labels[:,0]), LEN_BS)
        # time_grid_val = np.linspace(np.min(val_labels[:,0]), np.max(val_labels[:,0]), LEN_BS)
        # time_grid_test = np.linspace(np.min(test_labels[:,0]), np.max(test_labels[:,0]), LEN_BS)
        
        # Splitting the data:
        if(model_type not in _parameters.GNN_list):             
            if(_parameters.loss_fct in _parameters.survival_list): 
                train_data, val_data, test_data, input_model_size, clin_good_stand = split_data(model_type, train_list, val_list, test_list, train_labels, val_labels, test_labels, classical, list_centers, radiomics, bonus_array, clinical_array, clin_not_stand, ids, labels, Y, parameters_list, _parameters, edge_weights, train_labs[:,0], val_labs[:,0], test_labs[:,0])
            else:
                train_data, val_data, test_data, input_model_size, _ = split_data(model_type, train_list, val_list, test_list, train_labels, val_labels, test_labels, classical, list_centers, radiomics, bonus_array, clinical_array, clin_not_stand, ids, labels, Y, parameters_list, _parameters, edge_weights)
        if(_parameters.DB == 'METABRIC' and _parameters.loss_fct not in _parameters.survival_list): train_data, val_data = [train_data[i] for i in range(len(train_data)) if train_mask_nocens_before_mediantime[i]], [val_data[i] for i in range(len(val_data)) if val_mask_nocens_before_mediantime[i]]

        # Balance the data in the validation and testing sets + determine class weights for training:
        if((_parameters.loss_fct not in _parameters.survival_list) and (model_type not in _parameters.GNN_list)): val_neg, val_pos, test_neg, test_pos, weight_train = balance_data(train_labels, val_data, val_labels, test_data, test_labels) 
            

        # If we're on a cluster, we only do one loop of grid search per job:
        for loop_gridsearch in range(n_loops_gsearch):
            if(args.cluster): 
                loop_gridsearch, config_gridsearch = 0, int(args.grid_search_loop)
            else:
                config_gridsearch = loop_gridsearch

            if(_parameters.saving_results): writer = SummaryWriter(log_dir=log_dir+str(list(grid_param[config_gridsearch].items()))+'_Tloop_'+str(test_s+1)+'_loop_'+str(l+1))
            len_early_stopping = grid_param[config_gridsearch]["nb_epochs"]/2

            # If alpha varies in the grid search, we have to compute the graphs for each value of alpha:
            if (model_type in _parameters.GNN_list and _parameters.DB!='METABRIC' and _parameters.DB!='SUPPORT'):# and (not(Graphs_created))): 
                # # Graphs construction:
                # G, Graphs = graph_loop(len(Y), classical, list_centers, radiomics, Y, grid_param[config_gridsearch]["alpha"], _parameters) #Graphs is an array of graphs
                # Splitting them into the right sets:
                ind_alpha=np.where(np.array(parameters_list['alpha'])==grid_param[config_gridsearch]["alpha"])[0][0]
                train_data = [graphs[ind_alpha][index] for index in train_list]
                val_data = [graphs[ind_alpha][index] for index in val_list]
                test_data = [graphs[ind_alpha][index] for index in test_list]   
                print(graphs[0][0])
                input_model_size = graphs[0][0].x.shape[1]
                # Balance the data in the validation and testing sets + determine class weights for training:
                if(_parameters.loss_fct not in _parameters.survival_list): val_neg, val_pos, test_neg, test_pos, weight_train = balance_data(train_labels, val_data, val_labels, test_data, test_labels) 

            # Training dataset and model definition:
            train_loader = DataLoader(train_data, grid_param[config_gridsearch]["batch_size"], shuffle=True, collate_fn=collate) if (model_type in _parameters.MIL_list) else DataLoader(train_data, grid_param[config_gridsearch]["batch_size"], shuffle=True)
            if(_parameters.model == 'RSF_METABRIC'):
                n_estimators, min_samples_split, min_samples_leaf = grid_param[config_gridsearch]["n_estimators"], grid_param[config_gridsearch]["min_samples_split"], grid_param[config_gridsearch]["min_samples_leaf"]
                l1, C, HC = None, None, None
            else:
                n_estimators, min_samples_split, min_samples_leaf = None, None, None
                if(model_type=='LogReg_clin'):
                    l1, C = grid_param[config_gridsearch]["l1_ratio"], grid_param[config_gridsearch]["C_opt"]
                    HC = None
                else:
                    HC = grid_param[config_gridsearch]["hidden_channels"]
                    l1, C = None, None
            if(_parameters.loss_fct=='CoxTime' and loop_gridsearch==0): input_model_size += 1 #If we have CoxTime, we add an input for the time

            model = generate_model(model_type, HC, edge_weights, _parameters.sep_edge_weights, l1, C, _parameters.MLP2lay, _parameters.GNN1lay, input_model_size, _parameters.loss_fct, n_estimators, min_samples_split, min_samples_leaf, _parameters)


            # Loss and training parameters definition:
            class_weight = torch.tensor([weight_train], dtype=torch.float, requires_grad=False) if(_parameters.loss_fct not in _parameters.survival_list) else None           
            if(_parameters.loss_fct=='BCE'):
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weight)
            elif(_parameters.loss_fct=='DeepHitS'):
                criterion = pycox.models.loss.DeepHitSingleLoss(alpha=0.2, sigma=0.1)
            elif(_parameters.loss_fct=='CoxTime'):
                criterion = pycox.models.loss.CoxCCLoss(shrink=0.)
            elif(_parameters.loss_fct=='DeepSurv'):
                criterion = pycox.models.loss.CoxPHLoss()
            else:
                raise ValueError("_parameters.loss_function not implemented")
            model_surv = model #DeepHitSingle(model, optimizer, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts) if(_parameters.loss_fct=='DeepHitS') else None
            if(model_type == 'LogReg_clin'):
                optimizer, lr_scheduler = None, None
            elif(_parameters.model != 'RSF_METABRIC'):
                optimizer = torch.optim.Adam(model.parameters(), lr=grid_param[config_gridsearch]["learning_rates"])
                lr_scheduler = LRScheduler(optimizer, patience=5, min_lr=1e-6, factor=0.5)  
            if(_parameters.loss_fct=='CoxTime' and _parameters.model != 'RSF_METABRIC'): model_surv = CoxTime(model, tt.optim.Adam, labtrans=labtrans)
            if(_parameters.loss_fct=='DeepSurv' and _parameters.model != 'RSF_METABRIC'): model_surv = CoxPH(model, tt.optim.Adam)


            # Variables definition for early stopping:
            cntr_early_stopping = 0
            value_early_stop_max_run = -1
            early_stop = False


            #Specific process for RSF_METABRIC
            if(_parameters.model=='RSF_METABRIC'):
                #Train (no Validation) here (fit)
                print(f'Test loop {test_s+1:01d}, Loop {l+1:02d}, Loop grid search {config_gridsearch+1:02d}')
                #print(type(train_labels), train_labels, np.array([tuple(i) for i in np.flip(train_labels,1)], dtype=[('event','bool'),('event_time','float')]))
                print(datetime.datetime.today())
                model.fit(train_data, np.array([tuple(i) for i in np.flip(train_labels,1)], dtype=[('event','bool'),('event_time','float')])) #initial labels: list of 2 ndarrays [141., 39., ...] [1 0 ... 0]  #original train_data: ndarray (n_pat, n_feat)
                print(datetime.datetime.today())

                #Train metrics computation
                surv_array = model.predict_survival_function(train_data, return_array=True)
                surv = pandas.DataFrame(surv_array)
                surv.columns = model.unique_times_ #[f'pred_time_{i}' for i in range(len(surv.iloc[0]))]
                # print(surv) #n_pat rows and n_times columns
                ev = EvalSurv(surv.transpose(), train_labels[:,0], train_labels[:,1], censor_surv='km') #durations, events
                time_grid = np.linspace(train_labels[:,0].min(), train_labels[:,0].max(), 100)
                # bs[test_s, l, :] = ev.brier_score(time_grid)
                metrics_train[test_s, l, loop_gridsearch, :] = 0, ev.concordance_td(), ev.integrated_brier_score(time_grid)  #loss, c-index, IBS #no loss computation for the moment
                print(metrics_train[test_s, l, loop_gridsearch, 0])

                #Test
                surv_array = model.predict_survival_function(test_data, return_array=True)
                surv = pandas.DataFrame(surv_array)
                surv.columns = model.unique_times_ #[f'pred_time_{i}' for i in range(len(surv.iloc[0]))]
                preds_test = surv.copy()
                ev = EvalSurv(surv.transpose(), test_labels[:,0], test_labels[:,1], censor_surv='km')
                time_grid = np.linspace(test_labels[:,0].min(), test_labels[:,0].max(), 100)
                bs[test_s, l, :] = ev.brier_score(time_grid)
                metrics_test[test_s, l, :] = 0, ev.concordance_td(), ev.integrated_brier_score(time_grid)  #loss, c-index, IBS #no loss computation for the moment
                print(f'Test loop {test_s+1:01d}, Loop {l+1:02d}: Test IBS = {metrics_test[test_s, l, 2]:.4f}, Test c-index = {metrics_test[test_s, l, 1]:.4f}')

                ###################################################################################################################################
                #FOR EPN:
                if(_parameters.DB=='METABRIC' or _parameters.DB=='SUPPORT'):
                    list_EPN = []
                    #Get and standardize all data at once:
                    if(_parameters.DB=='METABRIC'):
                        cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
                        cols_leave = ['x4', 'x5', 'x6', 'x7'] #binary variables
                        all_cols = [f'x{i}' for i in range(len(cols_standardize+cols_leave))]
                    elif(_parameters.DB):
                        cols_standardize = ['x0', 'x2', 'x3', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']
                        cols_leave = ['x1', 'x4', 'x5'] #binary variables
                        all_cols = [f'x{i}' for i in range(len(cols_standardize+cols_leave))]
                    standardize = [([col], StandardScaler()) for col in cols_standardize]
                    leave = [(col, None) for col in cols_leave]
                    x_mapper = DataFrameMapper(standardize + leave, df_out=True)
                    all_feat_stand = np.array(x_mapper.fit_transform(bonus_array)[all_cols]).astype('float32')

                    get = [(col, None) for col in cols_leave+cols_standardize]
                    x_mapper2 = DataFrameMapper(get, df_out=True)
                    all_feat_nostand = np.array(x_mapper2.fit_transform(bonus_array)[all_cols]).astype('float32')

            
                    #Compute prediction for all data samples:
                    surv_array = model.predict_survival_function(all_feat_stand, return_array=True)
                    surv = pandas.DataFrame(surv_array)
                    surv.columns = model.unique_times_ #[f'pred_time_{i}' for i in range(len(surv.iloc[0]))]
                    preds_all = surv.copy()

                    if(len(grid_param)==1 and _parameters.saving_results):
                        #Get a vector identifying the split of each patient:
                        splits = []
                        for i in range(len(Y)):
                            if(i in train_list):
                                splits.append('train')
                            elif(i in val_list):
                                splits.append('val')
                            elif(i in test_list):
                                splits.append('test')
                            else:
                                raise ValueError(f'patient number {i} not in any list (train/val/test)')

                        #Save everything for the EPN:
                        surv = surv
                        for feat_nostand in range(len(all_feat_stand[0])):
                            surv[f'feat_{feat_nostand}'] = all_feat_nostand[:,feat_nostand]
                        surv['split'] = splits
                        surv['label_duration'] = Y[:,0]
                        surv['label_event'] = Y[:,1]
                        surv['pat_id'] = surv.index
                        #print(surv.head())                   
                        
                        surv.to_csv(f'./save/results/{_parameters.model}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/res_EPN_full_tloop_{test_s+1}_loop_{l+1}.csv', index=False)#, header=header_csv)
                
                
                #KM_curves:
                if(not _parameters.saving_results): writer=None
                y_all = Y.transpose()[0], Y.transpose()[1]
                pval_test_eventtime, pval_test_AUSC = kaplanmeyer_curves(y_test, preds_test, writer, test_s, l, len(grid_param), test_or_all_pat='test')
                pval_all_eventtime, pval_all_AUSC = kaplanmeyer_curves(y_all, preds_all, writer, test_s, l, len(grid_param), test_or_all_pat='allpat')
                p_values[test_s, l, loop_gridsearch, :] = pval_test_eventtime, pval_test_AUSC, pval_all_eventtime, pval_all_AUSC   
                
                #EXPORT FOR EPN
                np.save(f'./save/results/{_parameters.model}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/bs.npy', bs)#, header=header_csv)
                np.save(f'./save/results/{_parameters.model}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/timegrid.npy', time_grid_test_cont)#, header=header_csv)

                # TO EXPORT SPLITS
                np.save(f'./save/results/{_parameters.model}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/splits/train_split_tloop{test_s+1}_loop{l+1}.npy', train_list)
                np.save(f'./save/results/{_parameters.model}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/splits/val_split_tloop{test_s+1}_loop{l+1}.npy', val_list)
                np.save(f'./save/results/{_parameters.model}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/splits/test_split_tloop{test_s+1}_loop{l+1}.npy', test_list)
                ###################################################################################################################################


            #Specific process for CoxTime
            elif(_parameters.loss_fct=='CoxTime' or _parameters.loss_fct=='DeepSurv'):
                #Train + Validation here (fit)
                print(f'Test loop {test_s+1:01d}, Loop {l+1:02d}, Loop grid search {config_gridsearch+1:02d}')
                callbacks = [tt.callbacks.EarlyStopping()]
                model_surv.optimizer.set_lr(grid_param[config_gridsearch]["learning_rates"])

                # lrfinder = model_surv.lr_finder(train_data.astype('float32'), train_labels, batch_size=256, tolerance=10)
                # print(lrfinder.get_best_lr())
                # _ = lrfinder.plot()

                val = (val_data.astype('float32'), val_labels) if(_parameters.loss_fct == 'DeepSurv') else tt.tuplefy(val_data.astype('float32'), val_labels).repeat(100).cat()
                log = model_surv.fit(train_data.astype('float32'), train_labels, 256, grid_param[config_gridsearch]["nb_epochs"], callbacks, verbose=True, val_data=val) #val.repeat(10).cat()) #256=batch size
                # _ = log.plot()

                #Train metrics
                _ = model_surv.compute_baseline_hazards()
                surv = model_surv.predict_surv_df(train_data.astype('float32'))
                ev = EvalSurv(surv, train_labels[0][:], train_labels[1][:], censor_surv='km')
                time_grid = np.linspace(train_labels[0][:].min(), train_labels[0][:].max(), 100)
                # bs[test_s, l, :] = ev.brier_score(time_grid)
                metrics_train[test_s, l, loop_gridsearch, :] = 0, ev.concordance_td(), ev.integrated_brier_score(time_grid)  #loss, c-index, IBS #no loss computation for the moment
                print(metrics_train[test_s, l, loop_gridsearch, 0])

                #Validation metrics
                # _ = model_surv.compute_baseline_hazards()
                surv = model_surv.predict_surv_df(val_data.astype('float32'))
                ev = EvalSurv(surv, val_labels[0][:], val_labels[1][:], censor_surv='km')
                time_grid = np.linspace(val_labels[0][:].min(), val_labels[0][:].max(), 100)
                # bs[test_s, l, :] = ev.brier_score(time_grid)
                metrics_val[test_s, l, loop_gridsearch, :] = 0, ev.concordance_td(), ev.integrated_brier_score(time_grid)  #loss, c-index, IBS #no loss computation for the moment
                print(metrics_val[test_s, l, loop_gridsearch, 0])

                #Test
                # _ = model_surv.compute_baseline_hazards()
                surv = model_surv.predict_surv_df(test_data.astype('float32'))
                # print(surv)
                preds_test = surv.copy().transpose()
                ev = EvalSurv(surv, test_labels[0][:], test_labels[1][:], censor_surv='km')
                time_grid = np.linspace(test_labels[0][:].min(), test_labels[0][:].max(), 100)
                bs[test_s, l, :] = ev.brier_score(time_grid)
                # analysis_bs(surv, test_labels, surv.transpose().columns)

                #Bonus test metrics
                    #name_bonus_metrics = ['cindex_uncensored', 'ibs_uncensored', 'ibs_censored']
                censored_mask = torch.tensor(test_labels[1]==0)
                uncensored_mask = torch.tensor(test_labels[1]==1)
                print(np.count_nonzero(censored_mask | uncensored_mask), np.count_nonzero(censored_mask & uncensored_mask))
                ev_uncensored = EvalSurv(surv.iloc[:,np.where(uncensored_mask)[0]], test_labels[0][uncensored_mask], test_labels[1][uncensored_mask], censor_surv='km')
                time_grid_uncensored = np.linspace(test_labels[0][uncensored_mask].min(), test_labels[0][uncensored_mask].max(), LEN_BS)
                metrics_bonus_test_saved[test_s, l, 0:2] = ev_uncensored.concordance_td(), ev_uncensored.integrated_brier_score(time_grid_uncensored) 

                ev_censored = EvalSurv(surv.iloc[:,np.where(censored_mask)[0]], test_labels[0][censored_mask], test_labels[1][censored_mask], censor_surv='km')
                time_grid_censored = np.linspace(test_labels[0][censored_mask].min(), test_labels[0][censored_mask].max(), LEN_BS)
                metrics_bonus_test_saved[test_s, l, 2] = ev_censored.integrated_brier_score(time_grid_censored) 


                if(len(grid_param)==1 and _parameters.saving_results):
                    if(_parameters.DB == 'METABRIC' or _parameters.DB=='SUPPORT'):
                        np.save(f'./save/results/{_parameters.DB}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/bs.npy', bs)#, header=header_csv)
                        np.save(f'./save/results/{_parameters.DB}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/timegrid.npy', time_grid_test_cont)#, header=header_csv)

                        # TO EXPORT SPLITS
                        np.save(f'./save/results/{_parameters.DB}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/splits/train_split_tloop{test_s+1}_loop{l+1}.npy', train_list)
                        np.save(f'./save/results/{_parameters.DB}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/splits/val_split_tloop{test_s+1}_loop{l+1}.npy', val_list)
                        np.save(f'./save/results/{_parameters.DB}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/splits/test_split_tloop{test_s+1}_loop{l+1}.npy', test_list)
                        
                    else:
                        np.save(f'./save/results/{model_type}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/bs.npy', bs)#, header=header_csv)
                        np.save(f'./save/results/{model_type}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/timegrid.npy', time_grid_test_cont)#, header=header_csv)
                metrics_test[test_s, l, :] = 0, ev.concordance_td(), ev.integrated_brier_score(time_grid)  #loss, c-index, IBS #no loss computation for the moment
                print(f'Test loop {test_s+1:01d}, Loop {l+1:02d}: Test IBS = {metrics_test[test_s, l, 2]:.4f}, Test c-index = {metrics_test[test_s, l, 1]:.4f}')

                ###################################################################################################################################
                #FOR EPN:
                if(_parameters.DB=='METABRIC' or _parameters.DB=='SUPPORT'):
                    list_EPN = []
                    #Get and standardize all data at once:
                    if(_parameters.DB=='METABRIC'):
                        cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
                        cols_leave = ['x4', 'x5', 'x6', 'x7'] #binary variables
                        all_cols = [f'x{i}' for i in range(len(cols_standardize+cols_leave))]
                    elif(_parameters.DB):
                        cols_standardize = ['x0', 'x2', 'x3', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']
                        cols_leave = ['x1', 'x4', 'x5'] #binary variables
                        all_cols = [f'x{i}' for i in range(len(cols_standardize+cols_leave))]
                    standardize = [([col], StandardScaler()) for col in cols_standardize]
                    leave = [(col, None) for col in cols_leave]
                    x_mapper = DataFrameMapper(standardize + leave, df_out=True)
                    all_feat_stand = np.array(x_mapper.fit_transform(bonus_array)[all_cols]).astype('float32')

                    get = [(col, None) for col in cols_leave+cols_standardize]
                    x_mapper2 = DataFrameMapper(get, df_out=True)
                    all_feat_nostand = np.array(x_mapper2.fit_transform(bonus_array)[all_cols]).astype('float32')

            
                    #Compute prediction for all data samples:
                    _ = model_surv.compute_baseline_hazards()
                    surv = model_surv.predict_surv_df(all_feat_stand)
                    preds_all = surv.copy().transpose()

                    if(len(grid_param)==1 and _parameters.saving_results):
                        #Get a vector identifying the split of each patient:
                        splits = []
                        for i in range(len(Y)):
                            if(i in train_list):
                                splits.append('train')
                            elif(i in val_list):
                                splits.append('val')
                            elif(i in test_list):
                                splits.append('test')
                            else:
                                raise ValueError(f'patient number {i} not in any list (train/val/test)')

                        #Save everything for the EPN:
                        surv = surv.transpose()
                        for feat_nostand in range(len(all_feat_stand[0])):
                            surv[f'feat_{feat_nostand}'] = all_feat_nostand[:,feat_nostand]
                        surv['split'] = splits
                        surv['label_duration'] = Y[:,0]
                        surv['label_event'] = Y[:,1]
                        surv['pat_id'] = surv.index
                        print(surv.head())                   
                        
                        surv.to_csv(f'./save/results/{_parameters.DB}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/res_EPN_full_tloop_{test_s+1}_loop_{l+1}.csv', index=False)#, header=header_csv)
                else:                    
                    #Compute prediction for all data samples:
                    _ = model_surv.compute_baseline_hazards()
                    assert (model_type=='MLP_clin'), 'only MLP_clin implemented for EPN .csv file saving'
                    surv = model_surv.predict_surv_df(clin_good_stand.astype('float32'))
                    preds_all = surv.copy().transpose()
                    print(preds_all)

                    if(len(grid_param)==1 and _parameters.saving_results):
                        #Get a vector identifying the split of each patient:
                        splits = []
                        for i in range(len(Y)):
                            if(i in train_list):
                                splits.append('train')
                            elif(i in val_list):
                                splits.append('val')
                            elif(i in test_list):
                                splits.append('test')
                            else:
                                raise ValueError(f'patient number {i} not in any list (train/val/test)')
                        
                        #Save everything for the EPN:
                        surv = surv.transpose()
                        surv['split'] = splits
                        surv['label_duration'] = Y[:,0]
                        surv['label_event'] = Y[:,1]
                        surv['pat_id'] = patient_list
                            
                        for ind in range(len(clinical_array[0])):
                            surv[f'clin_vect_{ind+1}'] = clin_not_stand[:,ind]
                            surv[f'clin_stand1_{ind+1}'] = clinical_array[:,ind]
                            surv[f'clin_stand2_{ind+1}'] = clin_good_stand[:,ind]
                        print(surv.head())         

                        surv.to_csv(f'./save/results/{model_type}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/res_EPN_full_tloop_{test_s+1}_loop_{l+1}.csv', index=False)#, header=header_csv)
                
                
                #KM_curves:
                if(not _parameters.saving_results): writer=None
                y_all = Y.transpose()[0], Y.transpose()[1]
                pval_test_eventtime, pval_test_AUSC = kaplanmeyer_curves(y_test, preds_test, writer, test_s, l, len(grid_param), test_or_all_pat='test')
                pval_all_eventtime, pval_all_AUSC = kaplanmeyer_curves(y_all, preds_all, writer, test_s, l, len(grid_param), test_or_all_pat='allpat')
                p_values[test_s, l, loop_gridsearch, :] = pval_test_eventtime, pval_test_AUSC, pval_all_eventtime, pval_all_AUSC   
                ###################################################################################################################################
            else:
                for epoch in range(grid_param[config_gridsearch]["nb_epochs"]):
                    if (not(early_stop) or (not(_parameters.early_stopping))):
                        # roc_auc_train, train_loss, sensitivity_train, specificity_train, accuracy_train, balanced_accuracy_train, preds_train, labs_train = 0,0,0,0,0,0,0,0
                        # Training:
                        if(_parameters.loss_fct not in _parameters.survival_list): 
                            roc_auc_train, train_loss, sensitivity_train, specificity_train, accuracy_train, balanced_accuracy_train, _, preds_train, labs_train = train_val_test(model_type, "train", train_loader, model, criterion, m, optimizer, lr_scheduler, train_data, train_labels)
                            metrics_train[test_s, l, loop_gridsearch, epoch, :] = train_loss, roc_auc_train, accuracy_train, sensitivity_train, specificity_train, balanced_accuracy_train
                            if(_parameters.saving_results and len(grid_param)==1): 
                                write(writer, metrics_train, metrics_val, test_s, l, loop_gridsearch, config_gridsearch, epoch, name_metrics, max_ep=0, grid_param=grid_param, model_type=model_type, operation="grad_weights", model=model)
                                writer.add_pr_curve(f'pr_curve-train/Tloop_{test_s+1}_loop_{l+1}', labs_train, preds_train, global_step=epoch, num_thresholds=127)
                        else:
                            train_loss, cindex_train, _, ibs_train, preds_train, labs_train = train_val_test_surv(model_type, "train", train_loader, model, criterion, m, optimizer, lr_scheduler, train_data, train_labels, time_limits=time_intervals_limits, time_grid=time_grid_train_cont)
                            metrics_train[test_s, l, loop_gridsearch, epoch, :] = train_loss, cindex_train, ibs_train
                            if(_parameters.saving_results and len(grid_param)==1): 
                                write(writer, metrics_train, metrics_val, test_s, l, loop_gridsearch, config_gridsearch, epoch, name_metrics, max_ep=0, grid_param=grid_param, model_type=model_type, operation="grad_weights", model=model)    


                        # Validation (on N balanced sets if not a survival loss function):
                        if(_parameters.loss_fct not in _parameters.survival_list): 
                            sum_metrics = np.zeros(nb_metrics)
                            preds_val, labs_val = [],[]
                            for i in range(len(val_neg)):
                                val_data_split =[*val_pos, *val_neg[i]]
                                random.shuffle(val_data_split)
                                validation_loader = DataLoader(val_data_split, grid_param[config_gridsearch]["batch_size"], shuffle=True, collate_fn=collate) if (model_type in _parameters.MIL_list) else DataLoader(val_data_split, grid_param[config_gridsearch]["batch_size"], shuffle=True)                
                                roc_auc_val, val_loss, sensitivity_val, specificity_val, accuracy_val, balanced_accuracy_val, _, preds_val_tmp, labs_val_tmp = train_val_test(model_type, "val", validation_loader, model, criterion, m, data=val_data_split)
                                sum_metrics = np.add(sum_metrics, [val_loss,roc_auc_val,accuracy_val,sensitivity_val,specificity_val,balanced_accuracy_val])
                                labs_val.extend(labs_val_tmp)
                                preds_val.extend(preds_val_tmp)
                            metrics_val[test_s, l, loop_gridsearch, epoch, :] = sum_metrics/len(val_neg)
                            roc_auc_val_tot = metrics_val[test_s, l, loop_gridsearch, epoch, 1]
                            if(_parameters.saving_results and len(grid_param)==1): 
                                write(writer, metrics_train, metrics_val, test_s, l, loop_gridsearch, config_gridsearch, epoch, name_metrics, max_ep=0, grid_param=grid_param, model_type=model_type, model=model)
                                writer.add_pr_curve(f'pr_curve-val/Tloop_{test_s+1}_loop_{l+1}', np.array(labs_val), np.squeeze(preds_val), global_step=epoch, num_thresholds=127)
                            print(f'Test loop {test_s+1:01d}, Loop {l+1:02d}, Loop grid search {config_gridsearch+1:02d}, Epoch: {epoch:03d}: Train Bal Acc= {balanced_accuracy_train:.4f}, Val. Bal Acc= {metrics_val[test_s, l, loop_gridsearch, epoch, 5]:.4f}, Val. AU-ROC= {roc_auc_val_tot:.3f}')
                        else:
                            validation_loader = DataLoader(val_data, _parameters.batch_size, shuffle=True, collate_fn=collate) if (model_type in _parameters.MIL_list) else DataLoader(val_data, _parameters.batch_size, shuffle=True)                
                            val_loss, cindex_val, _, ibs_val, preds_val, labs_val  = train_val_test_surv(model_type, "val", validation_loader, model, criterion, m, data=val_data, time_limits=time_intervals_limits, time_grid=time_grid_val_cont)
                            metrics_val[test_s, l, loop_gridsearch, epoch, :] = val_loss, cindex_val, ibs_val
                            if(_parameters.saving_results and len(grid_param)==1): 
                                write(writer, metrics_train, metrics_val, test_s, l, loop_gridsearch, config_gridsearch, epoch, name_metrics, max_ep=0, grid_param=grid_param, model_type=model_type, model=model)
                            print(f'Test loop {test_s+1:01d}, Loop {l+1:02d}, Loop grid search {config_gridsearch+1:02d}, Epoch: {epoch:03d}: Train IBS= {ibs_train:.4f}, Val. IBS= {ibs_val:.4f}, Val. c-index= {cindex_val:.3f}, Train loss= {train_loss:.3f}')
                        
                    
                        
                        # Checking possible overfitting:
                        if(_parameters.loss_fct not in _parameters.survival_list): 
                            value_early_stop = roc_auc_val_tot
                        else:
                            value_early_stop = 1-ibs_val  #Because the value is between 0 and 1, and the lower the value the better
                        
                        if (value_early_stop <= value_early_stop_max_run):
                            cntr_early_stopping+=1 
                            if(cntr_early_stopping>=len_early_stopping):
                                early_stop = True
                                print(f"Early stopping: AU-ROC (for validation set) hasn't improved for {len_early_stopping} epochs.")
                        else:
                            cntr_early_stopping = 0
                            value_early_stop_max_run = value_early_stop
                            max_ep = epoch
                            if (value_early_stop > value_early_stop_max):
                                value_early_stop_max = value_early_stop
                                torch.save(model, f"./save/best_model_{model_type}.pth")
                                if(len(best_parameters_list)>l): del best_parameters_list[l]
                                best_parameters_list.append(grid_param[config_gridsearch])

            if(_parameters.saving_results and _parameters.loss_fct not in _parameters.survival_list): 
                write(writer, metrics_train, metrics_val, test_s, l, loop_gridsearch, config_gridsearch, epoch, name_metrics, max_ep, grid_param, model_type, operation="hparams", model=model)
                writer.flush()

            
        # Test:
        if(len(grid_param)==1 and _parameters.loss_fct != 'CoxTime' and _parameters.loss_fct != 'DeepSurv'):
            model = torch.load(f"./save/best_model_{model_type}.pth") #we retrieve the best model
            # if(model_type in _parameters.GNN_list): #if needed, we recompute the graphs with the best value of alpha       #Done earlier in fact
            #     #if(len(parameters_list["alpha"])>1 and _parameters.edge_weights_GNN):
            #         # Graphs construction:
            #         G, Graphs = graph_loop(len(Y), classical, list_centers, radiomics, Y, best_parameters_list[l]["alpha"], _parameters)
            #         # Balance the data in the testing set
            #         _, _, test_neg, test_pos, _ = balance_data(train_labels, val_data, val_labels, test_data=[Graphs[index] for index in test_list], test_labels=test_labels) 

            if(_parameters.loss_fct not in _parameters.survival_list): 
                sum_metrics = np.zeros(nb_metrics)
                preds_test, labs_test = [],[]
                for i in range(len(test_neg)):
                    test_data_split =[*test_pos, *test_neg[i]]
                    # print(test_data_split)
                    # print()
                    random.shuffle(test_data_split)
                    test_loader = DataLoader(test_data_split, batch_size=best_parameters_list[l]["batch_size"], shuffle=True, collate_fn=collate) if (model_type in _parameters.MIL_list) else DataLoader(test_data_split, batch_size=int(best_parameters_list[l]["batch_size"]), shuffle=True)
                    roc_auc_test, test_loss, sensitivity_test, specificity_test, accuracy_test, balanced_accuracy_test, confusion_mat_tmp, preds_test_tmp, labs_test_tmp  = train_val_test(model_type, "test", test_loader, model, criterion, m, data=test_data_split)
                    sum_metrics = np.add(sum_metrics, [test_loss,roc_auc_test,accuracy_test,sensitivity_test,specificity_test,balanced_accuracy_test])
                    confusion_mat[test_s, l] = np.add(confusion_mat[test_s, l],confusion_mat_tmp)
                    labs_test.extend(labs_test_tmp)
                    preds_test.extend(preds_test_tmp)
                metrics_test[test_s, l, :] = sum_metrics/len(test_neg)
                print(f'Test loop {test_s+1:01d}, Loop {l+1:02d}: Test Bal Acc = {metrics_test[test_s, l, 5]:.4f}, Test AU-ROC = {metrics_test[test_s, l, 1]:.4f}')    
            else:
                test_loader = DataLoader(test_data, batch_size=_parameters.batch_size, shuffle=True, collate_fn=collate) if (model_type in _parameters.MIL_list) else DataLoader(test_data, batch_size=_parameters.batch_size, shuffle=True)
                test_loss, cindex_test, bs_test, ibs_test, preds_test, labs_test  = train_val_test_surv(model_type, "test", test_loader, model, criterion, m, time_grid_test_cont[test_s], data=test_data, time_limits=time_intervals_limits, time_grid=time_grid_test_cont[test_s])
                metrics_test[test_s, l, :] = test_loss, cindex_test, ibs_test
                bs[test_s, l, :] = bs_test
                print(f'Test loop {test_s+1:01d}, Loop {l+1:02d}: Test IBS = {ibs_test:.4f}, Test c-index = {cindex_test:.4f}')

            # # TEST
            # test_data_split = [*test_pos, *test_neg[0]]
            # for i in range(len(test_neg)-1):
            #     test_data_split = [*test_pos, *test_neg[i+1], *test_data_split]
            # print(test_data_split)
            # random.shuffle(test_data_split)
            # test_loader = DataLoader(test_data_split, batch_size=best_parameters_list[l]["batch_size"], shuffle=True, collate_fn=collate) if (model_type in _parameters.MIL_list) else DataLoader(test_data_split, batch_size=int(best_parameters_list[l]["batch_size"]), shuffle=True)
            # roc_auc_test, test_loss, sensitivity_test, specificity_test, accuracy_test, balanced_accuracy_test, confusion_mat_test, preds_test, labs_test  = train_val_test(model_type, "test", test_loader, model, criterion, m, data=test_data_split)
            # confusion_mat[test_s, l, :,:] = confusion_mat_test
            # metrics_test[test_s, l, :] = sum_metrics/len(test_neg)

            if(_parameters.saving_results): 
                df = pandas.DataFrame(data=np.transpose(np.vstack((labs_test, np.squeeze(preds_test)))), columns=['test labels', 'test predictions']) if(_parameters.loss_fct not in _parameters.survival_list) \
                        else pandas.DataFrame(data=np.transpose(np.vstack((labs_test[0,:], labs_test[1,:], *[preds_test[:,i] for i in range(len(preds_test[0]))]))), columns=['durations test labels', 'events test labels', *[f'test predictions - time interval {i}' for i in range(len(preds_test[0]))]])
                df.to_csv(log_dir+'/../'+str(list(grid_param[config_gridsearch].items()))+f'_Tloop_{test_s+1}_loop_'+str(l+1)+'.csv')
                if(len(grid_param)==1 and _parameters.loss_fct not in _parameters.survival_list): writer.add_pr_curve(f'pr_curve-test/Tloop_{test_s+1}_loop_{l+1}', np.array(labs_test), np.squeeze(preds_test), global_step=epoch, num_thresholds=127)
            
            ###################################################################################################################################
            #FOR EPN:
            if(len(grid_param)==1):
                if(_parameters.DB=='METABRIC' or _parameters.DB=='SUPPORT'):
                    list_EPN = []
                    #Get and standardize all data at once:
                    if(_parameters.DB=='METABRIC'):
                        cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
                        cols_leave = ['x4', 'x5', 'x6', 'x7'] #binary variables
                        all_cols = [f'x{i}' for i in range(len(cols_standardize+cols_leave))]
                    elif(_parameters.DB):
                        cols_standardize = ['x0', 'x2', 'x3', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']
                        cols_leave = ['x1', 'x4', 'x5'] #binary variables
                        all_cols = [f'x{i}' for i in range(len(cols_standardize+cols_leave))]
                    standardize = [([col], StandardScaler()) for col in cols_standardize]
                    leave = [(col, None) for col in cols_leave]
                    x_mapper = DataFrameMapper(standardize + leave, df_out=True)
                    all_feat_stand = np.array(x_mapper.fit_transform(bonus_array)[all_cols]).astype('float32')

                    get = [(col, None) for col in cols_leave+cols_standardize]
                    x_mapper2 = DataFrameMapper(get, df_out=True)
                    all_feat_nostand = np.array(x_mapper2.fit_transform(bonus_array)[all_cols]).astype('float32')

                    list_EPN = []
                    model = torch.load(f"./save/best_model_{model_type}.pth") #we retrieve the best model of the loop 
                    print(len(train_mask_nocens_before_mediantime))
                    print(len(patient_list))
                    all_data = [{'feats': all_feat_stand[index], 'label': bin_Y[index,0]>mediantime, 'label_duration': bin_Y[index,0], 'label_event': bin_Y[index,1], \
                                 'train_mask_cens_before_mediantime': 1- train_mask_nocens_before_mediantime[np.where(train_list==index)[0][0]] if index in train_list else 0, 'val_mask_cens_before_mediantime': 1- val_mask_nocens_before_mediantime[np.where(val_list==index)[0][0]] if index in val_list else 0, \
                                 'pat_id': patient_list[index]}  for index in range(len(patient_list))] if(_parameters.loss_fct not in _parameters.survival_list)\
                        else [{'feats': all_feat_stand[index], 'label_pfs': Y[index,0], 'label_pfsevent': Y[index,1], 'pat_id': patient_list[index]}  for index in range(len(patient_list))]
                    full_loader = DataListLoader(all_data, shuffle=True)
                    
                    model.eval() #to set dropout and batch normalisation layers to evaluation mode
                    with torch.no_grad():
                        torch.manual_seed(42) #To be used #############################
                        for ind, data in enumerate(full_loader):
                            output = model(torch.tensor(data[0]['feats']).float(), None) #model(data[i]['graphs'].x, data[i]['graphs'].edge_index, data[i]['graphs'].edge_attr, data[i]['clin'])  # predict output from the model
                            # if (np.where(patient_list==data[i]['pat_id'])[0][0] in train_list):   #No need for the splits, and complex here
                            #     split = 'train'
                            # elif (np.where(patient_list==data[i]['pat_id'])[0][0] in val_list):
                            #     split = 'val'
                            # elif (np.where(patient_list==data[i]['pat_id'])[0][0] in test_list):                    
                            #     split = 'test'
                            # else:
                            #     raise AssertionError('patient_id not in any list (train/val/test)')

                            list_tmp = [int(data[0]['pat_id']), float(m(output)), data[0]['label'], float(data[0]['label_duration']), int(data[0]['label_event']), int(data[0]['train_mask_cens_before_mediantime']), int(data[0]['val_mask_cens_before_mediantime'])] if(_parameters.loss_fct not in _parameters.survival_list)\
                                else [int(data[0]['pat_id']), float(output), float(data[0]['label_duration']), int(data[0]['label_event'])]
                            list_tmp.extend(np.array(all_feat_stand[np.where(patient_list==data[0]['pat_id'])[0][0]], dtype=float))
                            list_tmp.extend(np.array(all_feat_nostand[np.where(patient_list==data[0]['pat_id'])[0][0]], dtype=float))
                            list_EPN.append(list_tmp)

                    # with open(f'./save/results/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/dict_EPN_tloop_{test_s+1}_loop_{l+1}.json', "w") as outfile: 
                    #     json.dump(dict_EPN, outfile)
                    df_EPN = pandas.DataFrame(list_EPN)
                    header_csv = ['pat_id', 'pred', 'label', 'label_duration', 'label_event', 'train_mask_cens_before_mediantime', 'val_mask_cens_before_mediantime'] if(_parameters.loss_fct not in _parameters.survival_list) else ['pat_id', 'labelpfs', 'label_pfsevent']  + [f'pred_time_{i}' for i in range(len(surv.iloc[0]))]
                    header_csv.extend([f'feat_stand_vect_{ind+1}' for ind in range(len(all_feat_stand[0]))])
                    header_csv.extend([f'feat_nostand_vect_{ind+1}' for ind in range(len(all_feat_stand[0]))])
            

                    # if(len(grid_param)==1 and _parameters.saving_results):
                        # #Get a vector identifying the split of each patient:
                        # splits = []
                        # for i in range(len(Y)):
                        #     if(i in train_list):
                        #         splits.append('train')
                        #     elif(i in val_list):
                        #         splits.append('val')
                        #     elif(i in test_list):
                        #         splits.append('test')
                        #     else:
                        #         raise ValueError(f'patient number {i} not in any list (train/val/test)')

                    df_EPN.to_csv(f'./save/results/{_parameters.DB}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/res_EPN_tloop_{test_s+1}_loop_{l+1}.csv', index=False, header=header_csv)
                
                    if(len(grid_param)==1 and _parameters.saving_results):
                        if(_parameters.DB == 'METABRIC' or _parameters.DB=='SUPPORT'):
                            # np.save(f'./save/results/{_parameters.DB}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/bs.npy', bs)#, header=header_csv)
                            # np.save(f'./save/results/{_parameters.DB}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/timegrid.npy', time_grid_test_cont)#, header=header_csv)

                            # TO EXPORT SPLITS
                            np.save(f'./save/results/{_parameters.DB}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/splits/train_split_tloop{test_s+1}_loop{l+1}.npy', train_list)
                            np.save(f'./save/results/{_parameters.DB}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/splits/val_split_tloop{test_s+1}_loop{l+1}.npy', val_list)
                            np.save(f'./save/results/{_parameters.DB}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/splits/test_split_tloop{test_s+1}_loop{l+1}.npy', test_list)
                        
                else:
                    list_EPN = []
                    model = torch.load(f"./save/best_model_{model_type}.pth") #we retrieve the best model of the loop 
                    all_data = [{'feats': feature_array[index], 'label': Y[index], 'pat_id': patient_list[index]}  for index in range(len(patient_list))] if(_parameters.loss_fct not in _parameters.survival_list)\
                        else [{'feats': feature_array[index], 'label_pfs': Y[index,0], 'label_pfsevent': Y[index,1], 'pat_id': patient_list[index]}  for index in range(len(patient_list))]
                    full_loader = DataListLoader(all_data, shuffle=True)
                    
                    model.eval() #to set dropout and batch normalisation layers to evaluation mode
                    with torch.no_grad():
                        torch.manual_seed(42) #To be used #############################
                        for ind, data in enumerate(full_loader):
                            output = model(torch.tensor(data[0]['feats']).float(), None) #model(data[i]['graphs'].x, data[i]['graphs'].edge_index, data[i]['graphs'].edge_attr, data[i]['clin'])  # predict output from the model
                            # if (np.where(patient_list==data[i]['pat_id'])[0][0] in train_list):   #No need for the splits, and complex here
                            #     split = 'train'
                            # elif (np.where(patient_list==data[i]['pat_id'])[0][0] in val_list):
                            #     split = 'val'
                            # elif (np.where(patient_list==data[i]['pat_id'])[0][0] in test_list):                    
                            #     split = 'test'
                            # else:
                            #     raise AssertionError('patient_id not in any list (train/val/test)')

                            list_tmp = [int(data[0]['pat_id']), float(m(output)), int(data[0]['label'])] if(_parameters.loss_fct not in _parameters.survival_list)\
                                else [int(data[0]['pat_id']), float(output), float(data[0]['label_duration']), int(data[0]['label_event'])]
                            list_tmp.extend(np.array(feature_array[np.where(patient_list==data[0]['pat_id'])[0][0]], dtype=float))
                            list_tmp.extend(np.array(clin_not_stand[np.where(patient_list==data[0]['pat_id'])[0][0]], dtype=float))
                            list_EPN.append(list_tmp)

                    # with open(f'./save/results/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/dict_EPN_tloop_{test_s+1}_loop_{l+1}.json', "w") as outfile: 
                    #     json.dump(dict_EPN, outfile)
                    df_EPN = pandas.DataFrame(list_EPN)
                    header_csv = ['pat_id', 'pred', 'label'] if(_parameters.loss_fct not in _parameters.survival_list) else ['pat_id', 'label_duration', 'label_event']  + [f'pred_time_{i}' for i in range(len(surv.iloc[0]))]
                    header_csv.extend([f'clin_stand_vect_{ind+1}' for ind in range(len(feature_array[0]))]) #clin vect of dim 10 
                    header_csv.extend([f'clin_nostand_vect_{ind+1}' for ind in range(len(feature_array[0]))]) #clin vect of dim 10 
            
                    # df_EPN.to_csv(f'./save/results/{model_type}/{date.year}_{date.month:02d}_{date.day:02d}-{date.hour:02d}h{date.minute:02d}_{date.second:02d}s{_parameters.name_save}/res_EPN_tloop_{test_s+1}_loop_{l+1}.csv', index=False, header=header_csv)
            ###################################################################################################################################


    if(_parameters.loss_fct not in _parameters.survival_list): 
        print(f'Test loop {test_s+1:01d}: Test Bal Acc = {np.mean(metrics_test[test_s, :, 5]):.4f}, Test AU-ROC = {np.mean(metrics_test[test_s, :, 1]):.4f}')
    else:
        print(f'Test loop {test_s+1:01d}: Test IBS = {np.mean(metrics_test[test_s, :, 2]):.4f}, Test c-index = {np.mean(metrics_test[test_s, :, 1]):.3f}')

# Identification of the best combination of parameters after grid search:
masked_metrics_val = np.ma.masked_equal(metrics_val, 0.0, copy=False) #Remove 0s from early stopping
if(_parameters.loss_fct not in _parameters.survival_list):
    mean_max_metric_val_all = np.mean(np.amax(metrics_val[:, :, :, :, 1], axis=3), axis=(0,1)) #returns the mean of the maximum value of the auroc for each loop for each combination of hyperparams.
    std_max_metric_val_all = np.std(np.amax(metrics_val[:, :, :, :, 1], axis=3), axis=(0,1)) #return the standard deviation of this value
else:
    mean_max_metric_val_all = np.mean(np.amin(masked_metrics_val[:, :, :, :, 1], axis=3), axis=(0,1)) #returns the mean of the minimum value of the c-index for each loop for each combination of hyperparams.
    std_max_metric_val_all = np.std(np.amin(masked_metrics_val[:, :, :, :, 1], axis=3), axis=(0,1)) #return the standard deviation of this value

# Displaying (& saving) results:
if(args.cluster): np.save(f"./tmp/{args.time_job}{_parameters.name_save}/res_{args.grid_search_loop}.npy", [mean_max_metric_val_all[0], std_max_metric_val_all[0]])
if(_parameters.saving_results):
    writer = SummaryWriter(log_dir=log_dir)
    if(args.cluster):        
        for i in range(max(parameters_list["nb_epochs"])):
            for m in range(len(name_metrics)):
                writer.add_scalars(f'{name_metrics[m]}-validation-Gsearch/{grid_param[config_gridsearch]}', {f'test_loop_{j+1}':np.mean(metrics_val[j, :, 0, i, m]) for j in range(len(metrics_test))}, i)
                writer.add_scalar(f'{name_metrics[m]}-validation-Gsearch/{grid_param[config_gridsearch]}/mean_value',np.mean(metrics_val[:, :, 0, i, m],axis=(0,1)), i)
                writer.add_scalar(f'{name_metrics[m]}-validation-Gsearch/{grid_param[config_gridsearch]}/std',np.std(metrics_val[:, :, 0, i, m],axis=(0,1)), i)
                writer.add_scalars(f'test/{name_metrics[m]}/', {f'test_loop_{j+1}':np.mean(metrics_test[j, :, m]) for j in range(len(metrics_test))})
                writer.add_scalar(f'test/{name_metrics[m]}/mean_value', np.mean(metrics_test[:, m], axis=(0,1)))
                writer.add_scalar(f'test/{name_metrics[m]}/std', np.std(metrics_test[:, :, m]))
    else:
        disp_save(model_type, parameters_list, metrics_train, metrics_val, metrics_test, confusion_mat, name_metrics, date, writer, grid_param, mean_max_metric_val_all, std_max_metric_val_all, pvalues=p_values, _parameters=_parameters)
    if(_parameters.loss_fct in _parameters.survival_list):
        for i in range(n_test_splits):
            fig = plt.figure() 
            ax = fig.add_subplot(111)
            ax.plot(time_grid_test_cont[i], np.mean(bs[i,:,:], axis=0))
            plt.title("Brier score")
            plt.xlabel("Time")
            writer.add_figure(f"_brier_scores/test_loop_{i+1}", fig, close=True)
    writer.flush()
    writer.close()
else:
    writer = None
    if(_parameters.loss_fct in _parameters.survival_list):
        for i in range(n_test_splits):
            fig = plt.figure() 
            ax = fig.add_subplot(111)
            ax.plot(time_grid_test_cont[i], np.mean(bs[i,:,:], axis=0))
            plt.title("Brier score")
            plt.xlabel("Time")
    if(not args.cluster): disp_save(model_type, parameters_list, metrics_train, metrics_val, metrics_test, confusion_mat, name_metrics, date, writer, grid_param, mean_max_metric_val_all, std_max_metric_val_all, save=False, pvalues=p_values, _parameters=_parameters)

for i in range(len(name_bonus_metrics)):
    print(name_bonus_metrics[i], np.mean(metrics_bonus_test_saved[:,:,i]), '+/-', np.std(metrics_bonus_test_saved[:,:,i]))
#mettre pvalue en val sur tb qd gsearch? (après je regarde plus les gsearch de ça sur tb)