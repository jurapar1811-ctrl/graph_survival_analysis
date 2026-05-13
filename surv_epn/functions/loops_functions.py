import sys
import copy
import torch
import pandas
import pickle
import datetime
import itertools
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from sklearn_pandas import DataFrameMapper
from pycox.datasets import metabric, support
from sksurv.ensemble import RandomSurvivalForest
from sklearn.linear_model import LogisticRegression
from pycox.models.cox_time import MLPVanillaCoxTime
from sklearn.preprocessing import StandardScaler, RobustScaler
###
sys.path.insert(1, './') 
from functions.models import MLP, GNN, GAT, XGBCoxTime
from functions.models import MILModel
from functions.MilDataset import MilDataset
from functions.graph_making import graph_loop
from functions.miscellaneous import get_graph, get_clinical, split_list, toTorchGeoData, select_best_epoch2, plot_training_curves2
from functions.getting_feat_new_DB import getting_feat_new_DB
from functions.train_test_functions import train_LogReg, test_LogReg, train_MLP, test_MLP, train_MIL, test_MIL, train_GNN, test_GNN, train_MLP_surv, test_MLP_surv
###
plt.close('all')
from main.parameters import parameters
_parameters = parameters()

LogReg_list = _parameters.LogReg_list
MLP_list = _parameters.MLP_list
MIL_list = _parameters.MIL_list
GNN_list = _parameters.GNN_list



# Variables initialization (by model):
def variables_init(model_type, _parameters, patient_list, Y, repo=''):  #Modified getting_feat_new_DB, only MLP_clin has been validated 
    lesion_array, ids, labels, classical, list_centers, radiomics, bonus_array, edge_weights, clinical_array, clin_not_stand = None, None, None, None, None, None, None, None, None, None

    if(_parameters.DB =='METABRIC' and _parameters.model != 'RSF_METABRIC'):
        # Get the learning parameters:
            parameters_list = _parameters.params_MLP_METABRIC_SUPPORT 

        # Get the data and standardize it:
            df_metabric = metabric.read_df()
            feature_array = np.arange(len(Y)) #=array of indices: the splits will just indicate which patient goes in each set
            bonus_array = df_metabric
            print(len(bonus_array))
            print(len(bonus_array.iloc[0]))

    elif(_parameters.DB =='METABRIC' and _parameters.model == 'RSF_METABRIC'):
        # Get the learning parameters:
            parameters_list = _parameters.params_RSF_METABRIC 

        # Get the data and standardize it:
            df_metabric = metabric.read_df()
            feature_array = np.arange(len(Y)) #=array of indices: the splits will just indicate which patient goes in each set
            bonus_array = df_metabric
            print(len(bonus_array))
            print(len(bonus_array.iloc[0]))

    elif(_parameters.DB =='SUPPORT'):
        # Get the learning parameters:
            parameters_list = _parameters.params_MLP_METABRIC_SUPPORT 

        # Get the data and standardize it:
            df_support = support.read_df()
            feature_array = np.arange(len(Y)) #=array of indices: the splits will just indicate which patient goes in each set
            bonus_array = df_support
            print(len(bonus_array))
            print(len(bonus_array.iloc[0]))

    else:    
        if (model_type == 'LogReg_clin'):
        # Get the learning parameters:
            parameters_list = _parameters.params_LogReg
            # patient_list = [str(x) for x in patient_list]
            # feature_array = get_clinical(patient_list, _parameters.clinical_data) #=clinical_features_array
            feature_array, _ = getting_feat_new_DB(_parameters, get_clin=True, get_img_feat=False, get_graphs=False, repo=repo)[0] #Won't work


        elif (model_type == 'MLP_clin'):
        # Get the learning parameters:
            parameters_list = _parameters.params_MLP 
        # Get the data:
            # patient_list = [str(x) for x in patient_list]
            # feature_array = get_clinical(patient_list, _parameters.clinical_data) #=clinical_features_array
            clinical_array, clin_not_stand = getting_feat_new_DB(_parameters, get_clin=True, get_img_feat=False, get_graphs=False, repo=repo)
            feature_array = np.arange(len(Y))


        elif (model_type == 'MLP_clin_graph'):
        # Get the learning parameters:
            parameters_list = _parameters.params_MLP 
        # Get the data:
            # with open(_parameters.PIK_graphs, "rb") as f:
            #     graphs = pickle.load(f)
            # with open(_parameters.PIK_dmax, "rb") as f:
            #     dmax = pickle.load(f)
            # graph_array = get_graph(graphs, dmax) #graph features
            # with open(_parameters.PIK_dump_graph, "wb") as f:
            #     pickle.dump(graph_array, f)
            # np.save(_parameters.PIK_dump_g_features, graph_array)

            # patient_list = [str(x) for x in patient_list]
            # clinical_array = get_clinical(patient_list, _parameters.clinical_data) #clinical features
            # feature_array = np.append(clinical_array, graph_array, axis=1) #=clinical+graph_features_array
            # bonus_array = graphs

            feat_array, _ = getting_feat_new_DB(_parameters, get_clin=True, get_img_feat=False, get_graphs=True, repo=repo)
            clinical_array, feature_array, bonus_array = feat_array[0], np.append(feat_array[0], feat_array[1], axis=1), feat_array[2]



        elif (model_type == 'MLP_clin_img'):
        # Get the learning parameters:
            parameters_list = _parameters.params_MLP 
        # Get the data:
            bonus_array = []
            # with open(_parameters.PIK_classical_data, "rb") as f:
            #     classical = pickle.load(f)
            # with open(_parameters.PIK_radiomics, "rb") as f:
            #     radiomics = pickle.load(f)
            # # One feature per patient (otherwise the number of features depends on the number of lesions of each patient and the MLP cannot deal with that)
            # for i in range(len(Y)):
            #     lesion_list = np.concatenate((classical[i], radiomics[i]), axis=1)
            #     bonus_array.append(np.mean(lesion_list, axis=0))

            # # feature_array = np.append(clinical_array, img_array, axis=1) #=clinical+image_features_array (clinical+classical+radiomics)
            # patient_list = [str(x) for x in patient_list]
            # clinical_array = get_clinical(patient_list, _parameters.clinical_data) #clinical features


            feat_array, _ = getting_feat_new_DB(_parameters, get_clin=True, get_img_feat=True, get_graphs=False, repo=repo)
            clinical_array, classical, radiomics = feat_array[0], feat_array[2], feat_array[3]
            # One feature per patient (otherwise the number of features depends on the number of lesions of each patient and the MLP cannot deal with that)
            for i in range(len(Y)):
                lesion_list = np.concatenate((classical[i], radiomics[i]), axis=1)
                bonus_array.append(np.mean(lesion_list, axis=0))
                
            feature_array = np.arange(len(Y)) #=array of indices: the splits will just indicate which patient goes in each set


        elif (model_type == 'MLP_clin_largest'):
        # Get the learning parameters:
            parameters_list = _parameters.params_MLP 
        # Get the data:
            feat_array, _ = getting_feat_new_DB(_parameters, get_clin=True, get_img_feat=True, get_graphs=False, repo=repo)
            clinical_array, classical, radiomics = feat_array[0], feat_array[2], feat_array[3]
            bonus_array = np.concatenate((classical, radiomics), axis=1)
                
            feature_array = np.arange(len(Y)) #=array of indices: the splits will just indicate which patient goes in each set


        elif (model_type == 'MLP_img'):
        # Get the learning parameters:
            parameters_list = _parameters.params_MLP 
        # Get the data:
            feature_array = []
            # with open(_parameters.PIK_classical_data, "rb") as f:
            #     classical = pickle.load(f)
            # with open(_parameters.PIK_radiomics, "rb") as f:
            #     radiomics = pickle.load(f)
            
            feat_array, _ = getting_feat_new_DB(_parameters, get_clin=False, get_img_feat=True, get_graphs=False, repo=repo)
            classical, radiomics = feat_array[1], feat_array[2]

            # One feature per patient (otherwise the number of features depends on the number of lesions of each patient and the MLP cannot deal with that)
            for i in range(len(Y)):
                lesion_list = np.concatenate((classical[i], radiomics[i]), axis=1)
                feature_array.append(np.mean(lesion_list, axis=0)) #=image_features_array (classical+radiomics)


        elif(model_type == 'MLP_largest'):
        # Get the learning parameters:
            parameters_list = _parameters.params_MLP 
        # Get the data:        
            feat_array, _ = getting_feat_new_DB(_parameters, get_clin=False, get_img_feat=True, get_graphs=False, repo=repo)
            classical, radiomics = feat_array[1], feat_array[2]
            feature_array = np.concatenate((classical, radiomics), axis=1)


        elif (model_type == 'MIL_img'):
        # Get the learning parameters:
            parameters_list = _parameters.params_MIL 
        # Get the data:
            # with open(_parameters.PIK_classical_data, "rb") as f:
            #     classical = pickle.load(f)
            # with open(_parameters.PIK_radiomics, "rb") as f:
            #     radiomics = pickle.load(f)
            feat_array, _ = getting_feat_new_DB(_parameters, get_clin=False, get_img_feat=True, get_graphs=False, repo=repo)
            classical, radiomics = feat_array[1], feat_array[2]

            lesion_array, ids = [],[]
            patient_id = list(range(len(Y)))
            for i in range(len(Y)):
                lesion_array.append(np.concatenate((classical[i], radiomics[i]), axis=1)) #each patient has n_lesions vectors composed of [classical,radiomics]
                ids.extend(list(itertools.repeat(patient_id[i], len(lesion_array[i])))) #returns a list: patient_id1{*nbROI_in_pat1}, patient_id2{*nbROI_in_pat2}, ...
            ids, labels = torch.tensor(ids), torch.tensor(Y)
            feature_array = np.arange(len(Y)) #=array of indices: the splits will just indicate which patient goes in each set
            bonus_array = lesion_array


        elif (model_type == 'GraphConv_img' or model_type == 'GAT_img'):
        # Get the learning parameters:
            parameters_list = _parameters.params_GNN 
            edge_weights = _parameters.edge_weights_GNN
        # Get the data:
            # with open(_parameters.PIK_classical_data, "rb") as f:
            #     classical = pickle.load(f)
            # with open(_parameters.PIK_center_tum, "rb") as f:
            #     list_centers = pickle.load(f)
            # with open(_parameters.PIK_radiomics, "rb") as f:
            #     radiomics = pickle.load(f)

            feat_array, _ = getting_feat_new_DB(_parameters, get_clin=False, get_img_feat=True, get_graphs=False, repo=repo)
            list_centers, classical, radiomics = feat_array[0], feat_array[1], feat_array[2]

            lesion_array = []
            for i in range(len(Y)):
                lesion_array.append(np.concatenate((classical[i], radiomics[i]), axis=1)) #each patient has n_lesions vectors composed of [classical,radiomics]
            feature_array = np.arange(len(Y)) #=array of indices: the splits will just indicate which patient goes in each set
            bonus_array = lesion_array


        else:
            raise ValueError("model_type is not in ['MLP_clin', 'MLP_clin_graph', 'MLP_img', 'MIL_img', 'GraphConv_img', 'GAT_img'] (in parameters.py)")

    return parameters_list, feature_array, bonus_array, ids, labels, classical, list_centers, radiomics, edge_weights, clinical_array, clin_not_stand
    



# Standardizing the data:
def split_data(model_type, train_list, val_list, test_list, train_labels, val_labels, test_labels, classical, list_centers, radiomics, bonus_array, clinical_array, clin_not_stand, ids, labels, Y, parameters_list, _parameters, edge_weights, train_cont_labels=None, val_cont_labels=None, test_cont_labels=None):
    clin_good_stand = None
    if(_parameters.loss_fct == 'BCE' and _parameters.DB == 'METABRIC'):
        train_data, val_data, test_data = [],[],[]
        cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
        cols_leave = ['x4', 'x5', 'x6', 'x7'] #binary variables
        all_cols = [f'x{i}' for i in range(len(cols_standardize+cols_leave))]

        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]

        x_mapper = DataFrameMapper(standardize + leave, df_out=True)
        train_list = np.array(x_mapper.fit_transform(bonus_array.iloc[train_list])[all_cols]).astype('float32')
        val_list = np.array(x_mapper.transform(bonus_array.iloc[val_list])[all_cols]).astype('float32')
        test_list = np.array(x_mapper.transform(bonus_array.iloc[test_list])[all_cols]).astype('float32')  

        for i in range(len(train_list)):
            train_data.append(toTorchGeoData([train_list[i].tolist()], train_labels[i]))
        for i in range(len(val_list)):
            val_data.append(toTorchGeoData([val_list[i].tolist()], val_labels[i]))
        for i in range(len(test_list)):
            test_data.append(toTorchGeoData([test_list[i].tolist()], test_labels[i])) 

        input_model_size = len(train_data[0].x[0]) #=9
        
    elif(_parameters.model == 'RSF_METABRIC' and _parameters.DB == 'METABRIC'):
            cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
            cols_leave = ['x4', 'x5', 'x6', 'x7'] #binary variables
            all_cols = [f'x{i}' for i in range(len(cols_standardize+cols_leave))]

            standardize = [([col], StandardScaler()) for col in cols_standardize]
            leave = [(col, None) for col in cols_leave]

            x_mapper = DataFrameMapper(standardize + leave, df_out=True)
            train_data = np.array(x_mapper.fit_transform(bonus_array.iloc[train_list])[all_cols]).astype('float32')
            val_data = np.array(x_mapper.transform(bonus_array.iloc[val_list])[all_cols]).astype('float32')
            test_data = np.array(x_mapper.transform(bonus_array.iloc[test_list])[all_cols]).astype('float32')    

            input_model_size = len(train_data[0]) #=9
        
    elif(_parameters.loss_fct != 'DeepSurv' and _parameters.loss_fct != 'CoxTime'):
        if(_parameters.DB == 'METABRIC'):
            train_data, val_data, test_data = [],[],[]
            cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
            cols_leave = ['x4', 'x5', 'x6', 'x7'] #binary variables
            all_cols = [f'x{i}' for i in range(len(cols_standardize+cols_leave))]

            standardize = [([col], StandardScaler()) for col in cols_standardize]
            leave = [(col, None) for col in cols_leave]

            x_mapper = DataFrameMapper(standardize + leave, df_out=True)
            train_list = np.array(x_mapper.fit_transform(bonus_array.iloc[train_list])[all_cols]).astype('float32')
            val_list = np.array(x_mapper.transform(bonus_array.iloc[val_list])[all_cols]).astype('float32')
            test_list = np.array(x_mapper.transform(bonus_array.iloc[test_list])[all_cols]).astype('float32')  

            for i in range(len(train_list)):
                train_data.append(toTorchGeoData([train_list[i].tolist()], train_labels[i], train_cont_labels[i]))
            for i in range(len(val_list)):
                val_data.append(toTorchGeoData([val_list[i].tolist()], val_labels[i], val_cont_labels[i]))
            for i in range(len(test_list)):
                test_data.append(toTorchGeoData([test_list[i].tolist()], test_labels[i], test_cont_labels[i])) 

            input_model_size = len(train_data[0]) #=9
        
        elif(model_type == 'LogReg_clin'):
            val_data, test_data = [],[]
            train_data = train_list
            for i in range(len(val_list)):
                val_data.append(toTorchGeoData([val_list[i].tolist()], val_labels[i], val_cont_labels[i]))
            for i in range(len(test_list)):
                test_data.append(toTorchGeoData([test_list[i].tolist()], test_labels[i], test_cont_labels[i]))
            input_model_size = None

        elif (model_type == 'MLP_clin_img' or model_type == 'MLP_clin_largest'):
            train_data, val_data, test_data = [],[],[]
            #Standardizing clinical data:
            clin_train_list = [clinical_array[index] for index in train_list]
            clin_val_list = [clinical_array[index] for index in val_list]
            clin_test_list = [clinical_array[index] for index in test_list]


            #Standardizing imaging data:
            lesion_tensor = torch.tensor(bonus_array, dtype=torch.float) #Standardization of all the data with the parameters of the training set

            for i in range(len(train_list)):
                train_data.append(toTorchGeoData([np.concatenate((clin_train_list[i], lesion_tensor[train_list[i]]), axis=0)], train_labels[i], train_cont_labels[i]))
            for i in range(len(val_list)):
                val_data.append(toTorchGeoData([np.concatenate((clin_val_list[i], lesion_tensor[val_list[i]]), axis=0)], val_labels[i], val_cont_labels[i]))
            for i in range(len(test_list)):
                test_data.append(toTorchGeoData([np.concatenate((clin_test_list[i], lesion_tensor[test_list[i]]), axis=0)], test_labels[i], test_cont_labels[i]))

            input_model_size = len(train_data[0].x[0]) #=25


        elif (model_type in MLP_list):
            train_data, val_data, test_data = [],[],[]
            if(model_type=='MLP_clin_graph'):
                for i in range(len(train_list)):
                    train_data.append(toTorchGeoData([train_list[i].tolist()], train_labels[i], bonus_array[i], train_cont_labels[i]))
                for i in range(len(val_list)):
                    val_data.append(toTorchGeoData([val_list[i].tolist()], val_labels[i], bonus_array[i], val_cont_labels[i]))
                for i in range(len(test_list)):
                    test_data.append(toTorchGeoData([test_list[i].tolist()], test_labels[i], bonus_array[i], test_cont_labels[i]))
            else:
                for i in range(len(train_list)):
                    train_data.append(toTorchGeoData([train_list[i].tolist()], train_labels[i], train_cont_labels[i]))
                for i in range(len(val_list)):
                    val_data.append(toTorchGeoData([val_list[i].tolist()], val_labels[i], val_cont_labels[i]))
                for i in range(len(test_list)):
                    test_data.append(toTorchGeoData([test_list[i].tolist()], test_labels[i], test_cont_labels[i]))

            input_model_size = len(train_data[0].x[0]) #=10 for MLP_clin, 13 for MLP_clin_graph and 15 for MLP_img


        elif (model_type in MIL_list): 
            lesion_list = []
            for i in range(len(Y)):
                for j in range(len(bonus_array[i])):
                    lesion_list.append(bonus_array[i][j]) #A list of lesions, with no distinction of patient
            lesion_tensor = torch.tensor(lesion_list, dtype=torch.float) #Standardization of all the data with the parameters of the training set
                    
            dataset = MilDataset(lesion_tensor, ids, labels)
            train_data = Subset(dataset, train_list)
            val_data = Subset(dataset, val_list) 
            test_data = Subset(dataset, test_list)

            input_model_size = len(dataset.data[0]) #=15


        elif (model_type in GNN_list):
            train_data = None
            val_data = None
            test_data = None

            input_model_size = None
            
        else:
            raise ValueError("model_type is not in any list (in parameters.py)")
        
    else: #survival case 
        if(_parameters.DB == 'METABRIC'):
            cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
            cols_leave = ['x4', 'x5', 'x6', 'x7'] #binary variables
            all_cols = [f'x{i}' for i in range(len(cols_standardize+cols_leave))]

            standardize = [([col], StandardScaler()) for col in cols_standardize]
            leave = [(col, None) for col in cols_leave]

            x_mapper = DataFrameMapper(standardize + leave, df_out=True)
            train_data = np.array(x_mapper.fit_transform(bonus_array.iloc[train_list])[all_cols]).astype('float32')
            val_data = np.array(x_mapper.transform(bonus_array.iloc[val_list])[all_cols]).astype('float32')
            test_data = np.array(x_mapper.transform(bonus_array.iloc[test_list])[all_cols]).astype('float32')    

            input_model_size = len(train_data[0]) #=9
        
        elif(_parameters.DB == 'SUPPORT'):   
            cols_standardize = ['x0', 'x2', 'x3', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']
            cols_leave = ['x1', 'x4', 'x5'] #binary variables
            all_cols = [f'x{i}' for i in range(len(cols_standardize+cols_leave))]

            standardize = [([col], StandardScaler()) for col in cols_standardize]
            leave = [(col, None) for col in cols_leave]

            x_mapper = DataFrameMapper(standardize + leave, df_out=True)
            train_data = np.array(x_mapper.fit_transform(bonus_array.iloc[train_list])[all_cols]).astype('float32')
            val_data = np.array(x_mapper.transform(bonus_array.iloc[val_list])[all_cols]).astype('float32')
            test_data = np.array(x_mapper.transform(bonus_array.iloc[test_list])[all_cols]).astype('float32')         

            input_model_size = len(train_data[0]) #=14

        else:
            if(model_type == 'LogReg_clin'):
                val_data, test_data = [],[]
                train_data = train_list
                for i in range(len(val_list)):
                    val_data.append(toTorchGeoData([val_list[i].tolist()], val_labels[0][i], val_labels[1][i]))
                for i in range(len(test_list)):
                    test_data.append(toTorchGeoData([test_list[i].tolist()], test_labels[i], test_cont_labels[i]))
                input_model_size = None

            elif (model_type == 'MLP_clin_img' or model_type == 'MLP_clin_largest'):
                train_data, val_data, test_data = [],[],[]
                #Standardizing clinical data:
                clin_train_list = [clinical_array[index] for index in train_list]
                clin_val_list = [clinical_array[index] for index in val_list]
                clin_test_list = [clinical_array[index] for index in test_list]


                #Standardizing imaging data:
                lesion_tensor = torch.tensor(bonus_array, dtype=torch.float) #Standardization of all the data with the parameters of the training set

                for i in range(len(train_list)):
                    train_data.append(([np.concatenate((clin_train_list[i], lesion_tensor[train_list[i]]), axis=0)], train_labels[i], train_cont_labels[i]))
                for i in range(len(val_list)):
                    val_data.append(([np.concatenate((clin_val_list[i], lesion_tensor[val_list[i]]), axis=0)], val_labels[0][i], val_labels[1][i]))
                for i in range(len(test_list)):
                    test_data.append(([np.concatenate((clin_test_list[i], lesion_tensor[test_list[i]]), axis=0)], test_labels[i], test_cont_labels[i]))

                input_model_size = len(train_data[0].x[0]) #=25


            elif (model_type in MLP_list):
                train_data, val_data, test_data = [],[],[]
                if(model_type=='MLP_clin_graph'):
                    for i in range(len(train_list)):
                        train_data.append(toTorchGeoData([train_list[i].tolist()], train_labels[i], bonus_array[i], train_cont_labels[i]))
                    for i in range(len(val_list)):
                        val_data.append(toTorchGeoData([val_list[i].tolist()], val_labels[i], bonus_array[i], val_cont_labels[i]))
                    for i in range(len(test_list)):
                        test_data.append(toTorchGeoData([test_list[i].tolist()], test_labels[i], bonus_array[i], test_cont_labels[i]))
                    input_model_size = len(train_data[0].x[0]) #=10 for MLP_clin, 13 for MLP_clin_graph and 15 for MLP_img
                else:
                    # for i in range(len(train_list)):
                    #     train_data.append(np.concatenate((train_list[i], train_labels[0][i], train_labels[1][i])))
                    # for i in range(len(val_list)):
                    #     val_data.append(np.concatenate((val_list[i], val_labels[0][i], val_labels[1][i])))
                    # for i in range(len(test_list)):
                    #     test_data.append(np.concatenate((test_list[i], test_labels[0][i], test_labels[1][i])))

                    # train_data, val_data, test_data = train_list, val_list, test_list


                    ################# FOR THE MLP_CLIN SURVIVAL ONLY (extend to others!!!), GOOD STANDARDIZATION OF CLINICAL DATA
                    scaler = StandardScaler()

                    train_data = np.array([clin_not_stand[index] for index in train_list])
                    scaler.fit(train_data)
                    clin_good_stand = scaler.transform(clin_not_stand)

                    train_data = np.array([clin_good_stand[index] for index in train_list])
                    val_data = np.array([clin_good_stand[index] for index in val_list])
                    test_data = np.array([clin_good_stand[index] for index in test_list])
                    input_model_size = len(train_data[0]) #=10 for MLP_clin, 13 for MLP_clin_graph and 15 for MLP_img


            elif (model_type in MIL_list): 
                lesion_list = []
                for i in range(len(Y)):
                    for j in range(len(bonus_array[i])):
                        lesion_list.append(bonus_array[i][j]) #A list of lesions, with no distinction of patient
                lesion_tensor = torch.tensor(lesion_list, dtype=torch.float) #Standardization of all the data with the parameters of the training set
                        
                dataset = MilDataset(lesion_tensor, ids, labels)
                train_data = Subset(dataset, train_list)
                val_data = Subset(dataset, val_list) 
                test_data = Subset(dataset, test_list)

                input_model_size = len(dataset.data[0]) #=15


            elif (model_type in GNN_list):
                train_data = None
                val_data = None
                test_data = None

                input_model_size = None
                
            else:
                raise ValueError("model_type is not in any list (in parameters.py)")

    return train_data, val_data, test_data, input_model_size, clin_good_stand


# Create a list of dictionaries with every configuration considered for grid search:
def gridsearch_params(parameters_list):
    param_values = []
    for _, values in parameters_list.items():
        param_values.append(values)
    grid_param = list(itertools.product(*param_values))
    gridsearch, dict_list = [],{}
    for i in range(len(grid_param)):
        j=0
        for names,_ in parameters_list.items():
            dict_list[names] = grid_param[i][j]
            j+=1
        gridsearch.append(dict_list.copy())
    return gridsearch


# Split the validation and test data to create balanced sets, and compute class weights for training:
def balance_data(train_labels, val_data, val_labels, test_data, test_labels):
    # Split negative class into equal-size lists:
    val_combined = list(zip(val_data, val_labels))
    test_combined = list(zip(test_data, test_labels))
    val_pos, val_neg, test_pos, test_neg = [],[],[],[]
    for i in range(len(val_combined)):
        if(val_combined[i][1] == 1):
            val_pos.append(val_data[i])
        else:
            val_neg.append(val_data[i])    
    for i in range(len(test_combined)):
        if(test_combined[i][1] == 1):
            test_pos.append(test_data[i])
        else:
            test_neg.append(test_data[i])
    val_neg = list(split_list(val_neg, 10))  #10 = len(val_pos)+1 (if we use 9=len(val_pos), the last set contains only 5 negative values)
    test_neg = list(split_list(test_neg, 10))  #10 = len(test_pos)+1 (if we use 9=len(test_pos), the last set contains only 4 negative values)

    # Get class weight:
    num_neg_train = train_labels.tolist().count(0)
    num_pos_train = train_labels.tolist().count(1)
    weight_train = num_neg_train / num_pos_train

    return val_neg, val_pos, test_neg, test_pos, weight_train


# Split the validation and test data to create balanced sets, and compute class weights for training:
def balance_data_EPN(train_labels, val_data, val_labels, test_data, test_labels):
    # Split negative class into equal-size lists:
    val_pos, val_neg, test_pos, test_neg = [],[],[],[]
    for i in range(len(val_labels)):
        if(val_labels[i] == 1):
            val_pos.append({'label': val_data[i]['label'], 'pat_id': val_data[i]['pat_id']})
        else:
            val_neg.append({'label': val_data[i]['label'], 'pat_id': val_data[i]['pat_id']})
    for i in range(len(test_labels)):
        if(test_labels[i] == 1):
            test_pos.append({'label': test_data[i]['label'], 'pat_id': test_data[i]['pat_id']})
        else:
            test_neg.append({'label': test_data[i]['label'], 'pat_id': test_data[i]['pat_id']})
    val_neg = list(split_list(val_neg, 10))  #10 = len(val_pos)+1 (if we use 9=len(val_pos), the last set contains only 5 negative values)
    test_neg = list(split_list(test_neg, 10))  #10 = len(test_pos)+1 (if we use 9=len(test_pos), the last set contains only 4 negative values)

    # Get class weight:
    num_neg_train = train_labels.tolist().count(0)
    num_pos_train = train_labels.tolist().count(1)
    weight_train = num_neg_train / num_pos_train

    return val_neg, val_pos, test_neg, test_pos, weight_train


# Generate the right model depending on model_type and its parameters:
def generate_model(model_type, hidden_channels, edge_weights, sep_edge_weights, l1, C, MLP2lay, GNN1lay, input_size=None, loss_fct=None, n_estimators=None, min_samples_split=None, min_samples_leaf=None, _parameters=None):
    torch.manual_seed(1258)
    output = _parameters.n_time_intervals if(_parameters.loss_fct == 'DeepHitS') else 1
    if(_parameters.DB == 'METABRIC' or _parameters.DB=='SUPPORT'):
        if(model_type == 'RSF_METABRIC'):
            model = RandomSurvivalForest(n_estimators=n_estimators, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, n_jobs=-1, random_state=42)
        elif(model_type == 'XGB_METABRIC'):
            model = XGBCoxTime()
        else:
            model = MLPVanillaCoxTime(input_size-1, [hidden_channels, hidden_channels], batch_norm=True, dropout=0.1) 

    elif (model_type in LogReg_list):
        model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=l1, C=C, n_jobs=-1, max_iter=200)
    elif (model_type in MLP_list):
        # CoxTime = True if(loss_fct=='CoxTime') else False
        model = MLP(hidden_channels, input_size, MLP2lay, output_size=output, CoxTime=(loss_fct=='CoxTime'))
        if(_parameters.loss_fct == 'CoxTime'):
            model = MLPVanillaCoxTime(input_size-1, [hidden_channels, hidden_channels], batch_norm=True, dropout=0.1)  #in fact use this one

    elif (model_type in MIL_list):
        prepNN = torch.nn.Sequential(
                    torch.nn.Linear(input_size, hidden_channels),   
                    torch.nn.ReLU())
        afterNN = torch.nn.Sequential(torch.nn.Linear(hidden_channels, 1))
        model = MILModel(prepNN, afterNN, torch.max)
    elif (model_type == 'GraphConv_img'):
        model = GNN(hidden_channels, input_size, edge_weights, sep_edge_weights, GNN1lay)
    elif (model_type == 'GAT_img'):
        model = GAT(hidden_channels, input_size, edge_weights, sep_edge_weights, GNN1lay)
    else:
        raise ValueError("model_type is not in LogReg_list, MLP_list, MIL_list, GNN_list, METsurv_list (in parameters.py)")
    return model


# Choose which function to use for training, validation or testing:
def train_val_test(model_type, operation, loader, model, criterion, m, optimizer=None, lr_scheduler=None, data=None, labels=None):
    if(operation=="train"):
        if (model_type in LogReg_list):
            bal_accuracy, roc_auc, sensitivity, specificity, accuracy, loss, cm, preds, labs = train_LogReg(data, labels, model)
        elif (model_type in MLP_list):
            bal_accuracy, roc_auc, sensitivity, specificity, accuracy, loss, cm, preds, labs = train_MLP(loader, model, optimizer, criterion, m, lr_scheduler)
        elif (model_type in MIL_list):
            bal_accuracy, roc_auc, sensitivity, specificity, accuracy, loss, cm, preds, labs = train_MIL(loader, model, optimizer, criterion, m, lr_scheduler)
        elif (model_type in GNN_list):
            bal_accuracy, roc_auc, sensitivity, specificity, accuracy, loss, cm, preds, labs = train_GNN(loader, model, optimizer, criterion, m, lr_scheduler)
        else:
            raise ValueError("model_type is not in any list (in parameters.py)")
    elif(operation=="val" or operation=="test"):
        if (model_type in LogReg_list):
            _, roc_auc, _, _, sensitivity, specificity, accuracy, bal_accuracy, loss, cm, preds, labs = test_LogReg(data, model)
        elif (model_type in MLP_list):
            _, roc_auc, _, _, sensitivity, specificity, accuracy, bal_accuracy, loss, cm, preds, labs = test_MLP(loader, model, m, criterion)
        elif (model_type in MIL_list):
            _, roc_auc, _, _, sensitivity, specificity, accuracy, bal_accuracy, loss, cm, preds, labs = test_MIL(loader, model, m, criterion)
        elif (model_type in GNN_list):
            _, roc_auc, _, _, sensitivity, specificity, accuracy, bal_accuracy, loss, cm, preds, labs = test_GNN(loader, model, m, criterion)
        else:
            raise ValueError("model_type is not in any list (in parameters.py)")
    else:
        raise ValueError("Problem: operation not in {'train', 'val', 'test'} (in function train_val_test)")
    return roc_auc, loss, sensitivity, specificity, accuracy, bal_accuracy, cm, preds, labs


# Send data to tensorboard
def write(writer, metrics_train, metrics_val, test_s, l, loop_gridsearch, config_gridsearch, epoch, name_metrics, max_ep, grid_param, model_type, operation="scalar", model=None):
    if(operation=="scalar"):
        for m in range(len(name_metrics)):
            writer.add_scalar(f'{name_metrics[m]}-train/Tloop{test_s+1}_loop_{l+1}', metrics_train[test_s, l, loop_gridsearch, epoch, m], epoch)
            writer.add_scalar(f'{name_metrics[m]}-validation/Tloop{test_s+1}_loop_{l+1}', metrics_val[test_s, l, loop_gridsearch, epoch, m], epoch)
    elif(operation=="grad_weights"):
        pass
        # if(len(grid_param)==1):
        #     writer.add_histogram(f'weights/conv1_att/loop_{l+1}', model.conv1.att, global_step=epoch)
        #     writer.add_histogram(f'weights/conv1_lin_edge/loop_{l+1}', model.conv1.lin_edge.weight, global_step=epoch)
        #     writer.add_histogram(f'weights/conv1_lin_l/loop_{l+1}', model.conv1.lin_l.weight, global_step=epoch)
        #     writer.add_histogram(f'weights/conv1_lin_r/loop_{l+1}', model.conv1.lin_r.weight, global_step=epoch)  
        #     writer.add_histogram(f'weights/att11_q_proj_weight/loop_{l+1}', model.att11.q_proj_weight, global_step=epoch)
        #     writer.add_histogram(f'weights/att11_k_proj_weight/loop_{l+1}', model.att11.k_proj_weight, global_step=epoch)
        #     writer.add_histogram(f'weights/att11_v_proj_weight/loop_{l+1}', model.att11.v_proj_weight, global_step=epoch)  
        #     writer.add_histogram(f'weights/conv2_att/loop_{l+1}', model.conv2.att, global_step=epoch)
        #     writer.add_histogram(f'weights/conv2_lin_edge/loop_{l+1}', model.conv2.lin_edge.weight, global_step=epoch)
        #     writer.add_histogram(f'weights/conv2_lin_l/loop_{l+1}', model.conv2.lin_l.weight, global_step=epoch)
        #     writer.add_histogram(f'weights/conv2_lin_r/loop_{l+1}', model.conv2.lin_r.weight, global_step=epoch)  
        #     writer.add_histogram(f'weights/att21_q_proj_weight/loop_{l+1}', model.att21.q_proj_weight, global_step=epoch)
        #     writer.add_histogram(f'weights/att21_k_proj_weight/loop_{l+1}', model.att21.k_proj_weight, global_step=epoch)
        #     writer.add_histogram(f'weights/att21_v_proj_weight/loop_{l+1}', model.att21.v_proj_weight, global_step=epoch)  
        #     writer.add_histogram(f'weights/lin_out/loop_{l+1}', model.lin_out.weight, global_step=epoch)   

        # writer.add_scalars(f'weights_grad/loop_{l+1}', {'conv1_att': torch.linalg.norm(model.conv1.att.grad), 
        #                                             'conv1_lin_edge': torch.linalg.norm(model.conv1.lin_edge.weight.grad),
        #                                             'conv1_lin_l': torch.linalg.norm(model.conv1.lin_l.weight.grad),
        #                                             'conv1_lin_r': torch.linalg.norm(model.conv1.lin_r.weight.grad),
        #                                             'att11_q_proj_weight': torch.linalg.norm(model.att11.q_proj_weight.grad),
        #                                             'att11_k_proj_weight': torch.linalg.norm(model.att11.k_proj_weight.grad),
        #                                             'att11_v_proj_weight': torch.linalg.norm(model.att11.v_proj_weight.grad),
        #                                             'conv2_att': torch.linalg.norm(model.conv2.att.grad), 
        #                                             'conv2_lin_edge': torch.linalg.norm(model.conv2.lin_edge.weight.grad),
        #                                             'conv2_lin_l': torch.linalg.norm(model.conv2.lin_l.weight.grad),
        #                                             'conv2_lin_r': torch.linalg.norm(model.conv2.lin_r.weight.grad),
        #                                             'att21_q_proj_weight': torch.linalg.norm(model.att21.q_proj_weight.grad),
        #                                             'att21_k_proj_weight': torch.linalg.norm(model.att21.k_proj_weight.grad),
        #                                             'att21_v_proj_weight': torch.linalg.norm(model.att21.v_proj_weight.grad),
        #                                             'lin_out': torch.linalg.norm(model.lin_out.weight.grad)}, epoch)
    else:
        if(model_type == 'LogReg_clin'):
            LR = grid_param[config_gridsearch]["l1_ratio"]
            HC = grid_param[config_gridsearch]["C_opt"]
        else:
            LR = grid_param[config_gridsearch]["learning_rates"]
            HC = grid_param[config_gridsearch]["hidden_channels"]
        writer.add_hparams({'batch_size':grid_param[config_gridsearch]["batch_size"], 'hidden_channels':HC,\
                            'learning_rate':LR, 'nb_epochs':grid_param[config_gridsearch]["nb_epochs"], \
                            'alpha':0 if (model_type not in _parameters.GNN_list) else grid_param[config_gridsearch]["alpha"], 'Tloop':test_s+1, 'loop':l+1}, \
                           {'_hparam/au-roc':metrics_val[test_s, l, loop_gridsearch, max_ep, 1], '_hparam/bal_acc_max_val':metrics_val[test_s, l, loop_gridsearch, max_ep, 5], \
                            '_hparam/accuracy':metrics_val[test_s, l, loop_gridsearch, max_ep, 2], '_hparam/loss':metrics_val[test_s, l, loop_gridsearch, max_ep, 0],\
                            '_hparam/sensitivity':metrics_val[test_s, l, loop_gridsearch, max_ep, 3], '_hparam/specificity':metrics_val[test_s, l, loop_gridsearch, max_ep, 4]})  
        writer.add_scalar('_best_epoch', max_ep)
    
    return 0


# Choose which function to use for training, validation or testing (with survival loss):
def train_val_test_surv(model_type, operation, loader, model, criterion, m, optimizer=None, lr_scheduler=None, data=None, labels=None, time_limits=None, time_grid=None):
    bs = None
    if(operation=="train"):
        if(model_type in MLP_list):
            loss, cindex, ibs, preds, labs = train_MLP_surv(loader, model, optimizer, criterion, lr_scheduler, time_limits, time_grid)
        else:
            raise ValueError("_parameters.loss_fct in _parameters.survival_list and model_type not in MLP_list: not yet implemented")

        # if (model_type in LogReg_list):
        #     bal_accuracy, roc_auc, sensitivity, specificity, accuracy, loss, cm, preds, labs = train_LogReg(data, labels, model)
        # elif (model_type in MLP_list):
        #     bal_accuracy, roc_auc, sensitivity, specificity, accuracy, loss, cm, preds, labs = train_MLP(loader, model, optimizer, criterion, m, lr_scheduler)
        # elif (model_type in MIL_list):
        #     bal_accuracy, roc_auc, sensitivity, specificity, accuracy, loss, cm, preds, labs = train_MIL(loader, model, optimizer, criterion, m, lr_scheduler)
        # elif (model_type in GNN_list):
        #     bal_accuracy, roc_auc, sensitivity, specificity, accuracy, loss, cm, preds, labs = train_GNN(loader, model, optimizer, criterion, m, lr_scheduler)
        # else:
        #     raise ValueError("model_type is not in any list (in parameters.py)")
    elif(operation=="val" or operation=="test"):
        if(model_type in MLP_list):
            loss, cindex, bs, ibs, preds, labs, metrics_bonus_test = test_MLP_surv(loader, model, criterion, time_limits, time_grid)
        else:
            raise ValueError("_parameters.loss_fct in _parameters.survival_list and model_type not in MLP_list: not yet implemented")
        
        # if (model_type in LogReg_list):
        #     _, roc_auc, _, _, sensitivity, specificity, accuracy, bal_accuracy, loss, cm, preds, labs = test_LogReg(data, model)
        # elif (model_type in MLP_list):
        #     _, roc_auc, _, _, sensitivity, specificity, accuracy, bal_accuracy, loss, cm, preds, labs = test_MLP(loader, model, m, criterion)
        # elif (model_type in MIL_list):
        #     _, roc_auc, _, _, sensitivity, specificity, accuracy, bal_accuracy, loss, cm, preds, labs = test_MIL(loader, model, m, criterion)
        # elif (model_type in GNN_list):
        #     _, roc_auc, _, _, sensitivity, specificity, accuracy, bal_accuracy, loss, cm, preds, labs = test_GNN(loader, model, m, criterion)
        # else:
        #     raise ValueError("model_type is not in any list (in parameters.py)")
    else:
        raise ValueError("Problem: operation not in {'train', 'val', 'test'} (in function train_val_test)")
    return loss, cindex, bs, ibs, preds, labs

# Send data to tensorboard
def write_EPN(writer, metrics_train, metrics_val, test_s, l, loop_gridsearch, config_gridsearch, epoch, name_metrics, max_ep, grid_param, operation="scalar", model=None):
    if(operation=="scalar"):
        for m in range(len(name_metrics)):
            writer.add_scalar(f'{name_metrics[m]}-train/Tloop{test_s+1}_loop_{l+1}', metrics_train[test_s, l, loop_gridsearch, epoch, m], epoch)
            writer.add_scalar(f'{name_metrics[m]}-validation/Tloop{test_s+1}_loop_{l+1}', metrics_val[test_s, l, loop_gridsearch, epoch, m], epoch)
    elif(operation=="grad_weights"):
        if(len(grid_param)==1):
            writer.add_histogram(f'weights/conv1_att/Tloop{test_s+1}_loop_{l+1}', model.conv1.att, global_step=epoch)
            writer.add_histogram(f'weights/conv1_lin_edge/Tloop{test_s+1}_loop_{l+1}', model.conv1.lin_edge.weight, global_step=epoch)
            writer.add_histogram(f'weights/conv1_lin_l/Tloop{test_s+1}_loop_{l+1}', model.conv1.lin_l.weight, global_step=epoch)
            writer.add_histogram(f'weights/conv1_lin_r/Tloop{test_s+1}_loop_{l+1}', model.conv1.lin_r.weight, global_step=epoch)  
            writer.add_histogram(f'weights/att11_q_proj_weight/Tloop{test_s+1}_loop_{l+1}', model.att11.q_proj_weight, global_step=epoch)
            writer.add_histogram(f'weights/att11_k_proj_weight/Tloop{test_s+1}_loop_{l+1}', model.att11.k_proj_weight, global_step=epoch)
            writer.add_histogram(f'weights/att11_v_proj_weight/Tloop{test_s+1}_loop_{l+1}', model.att11.v_proj_weight, global_step=epoch)  
            writer.add_histogram(f'weights/conv2_att/Tloop{test_s+1}_loop_{l+1}', model.conv2.att, global_step=epoch)
            writer.add_histogram(f'weights/conv2_lin_edge/Tloop{test_s+1}_loop_{l+1}', model.conv2.lin_edge.weight, global_step=epoch)
            writer.add_histogram(f'weights/conv2_lin_l/Tloop{test_s+1}_loop_{l+1}', model.conv2.lin_l.weight, global_step=epoch)
            writer.add_histogram(f'weights/conv2_lin_r/Tloop{test_s+1}_loop_{l+1}', model.conv2.lin_r.weight, global_step=epoch)  
            writer.add_histogram(f'weights/att21_q_proj_weight/Tloop{test_s+1}_loop_{l+1}', model.att21.q_proj_weight, global_step=epoch)
            writer.add_histogram(f'weights/att21_k_proj_weight/Tloop{test_s+1}_loop_{l+1}', model.att21.k_proj_weight, global_step=epoch)
            writer.add_histogram(f'weights/att21_v_proj_weight/Tloop{test_s+1}_loop_{l+1}', model.att21.v_proj_weight, global_step=epoch)  
            writer.add_histogram(f'weights/lin_out/Tloop{test_s+1}_loop_{l+1}', model.lin_out.weight, global_step=epoch)   

        writer.add_scalars(f'weights_grad/Tloop{test_s+1}_loop_{l+1}', {'conv1_att': torch.linalg.norm(model.conv1.att.grad), 
                                                                        'conv1_lin_edge': torch.linalg.norm(model.conv1.lin_edge.weight.grad),
                                                                        'conv1_lin_l': torch.linalg.norm(model.conv1.lin_l.weight.grad),
                                                                        'conv1_lin_r': torch.linalg.norm(model.conv1.lin_r.weight.grad),
                                                                        'att11_q_proj_weight': torch.linalg.norm(model.att11.q_proj_weight.grad),
                                                                        'att11_k_proj_weight': torch.linalg.norm(model.att11.k_proj_weight.grad),
                                                                        'att11_v_proj_weight': torch.linalg.norm(model.att11.v_proj_weight.grad),
                                                                        'conv2_att': torch.linalg.norm(model.conv2.att.grad), 
                                                                        'conv2_lin_edge': torch.linalg.norm(model.conv2.lin_edge.weight.grad),
                                                                        'conv2_lin_l': torch.linalg.norm(model.conv2.lin_l.weight.grad),
                                                                        'conv2_lin_r': torch.linalg.norm(model.conv2.lin_r.weight.grad),
                                                                        'att21_q_proj_weight': torch.linalg.norm(model.att21.q_proj_weight.grad),
                                                                        'att21_k_proj_weight': torch.linalg.norm(model.att21.k_proj_weight.grad),
                                                                        'att21_v_proj_weight': torch.linalg.norm(model.att21.v_proj_weight.grad),
                                                                        'lin_out': torch.linalg.norm(model.lin_out.weight.grad)}, epoch)
    elif(operation=="hparams_classif"):
        writer.add_hparams({'alpha':grid_param[config_gridsearch]["alpha"], 'beta':grid_param[config_gridsearch]["beta"],\
                            'kernel_att':grid_param[config_gridsearch]["kernel_att"], \
                            'learning_rate':grid_param[config_gridsearch]["LR"], 'prop_neighbors':grid_param[config_gridsearch]["prop_neighbors"],\
                            'nb_epochs':grid_param[config_gridsearch]["nb_epochs"], 'Tloop':test_s+1, 'loop':l+1}, \
                           {'_hparam/roc_auc':metrics_val[test_s, l, loop_gridsearch, max_ep, 1], '_hparam/bal_acc_max_val':metrics_val[test_s, l, loop_gridsearch, max_ep, 5], \
                            '_hparam/accuracy':metrics_val[test_s, l, loop_gridsearch, max_ep, 2], '_hparam/loss':metrics_val[test_s, l, loop_gridsearch, max_ep, 0],\
                            '_hparam/sensitivity':metrics_val[test_s, l, loop_gridsearch, max_ep, 3], '_hparam/specificity':metrics_val[test_s, l, loop_gridsearch, max_ep, 4]})
        if(len(grid_param)==1):
            writer.add_scalar('_best_epoch', max_ep)
    elif(operation=="hparams_survival"):
        writer.add_hparams({'alpha':grid_param[config_gridsearch]["alpha"], 'beta':grid_param[config_gridsearch]["beta"],\
                            'kernel_att':grid_param[config_gridsearch]["kernel_att"], \
                            'learning_rate':grid_param[config_gridsearch]["LR"], 'prop_neighbors':grid_param[config_gridsearch]["prop_neighbors"],\
                            'nb_epochs':grid_param[config_gridsearch]["nb_epochs"], 'Tloop':test_s+1, 'loop':l+1}, \
                           {'_hparam/IBS':metrics_val[test_s, l, loop_gridsearch, max_ep, 2], '_hparam/c_index':metrics_val[test_s, l, loop_gridsearch, max_ep, 1], \
                            '_hparam/loss':metrics_val[test_s, l, loop_gridsearch, max_ep, 0]})
        if(len(grid_param)==1):
            writer.add_scalar('_best_epoch', max_ep)
    else:
        raise ValueError('operation in write_EPN() does not match any implemented operation')
    writer.flush()


# Display (& save) the results:
def disp_save(model_type, parameters_list, metrics_train, metrics_val, metrics_test, confusion_mat, name_metrics, date_begin, writer, gsearch, mean_max_roc_auc_val, std_max_roc_auc_val, saving_repo=None, EPN=False, save=True, pvalues=None, _parameters=None):
    if not EPN:
        if((_parameters.DB=='METABRIC' or _parameters.DB=='SUPPORT') and _parameters.model!='RSF_METABRIC'): model_type = _parameters.DB
    repo_to_save = f'./save/results/{model_type}/{date_begin.year}_{date_begin.month:02d}_{date_begin.day:02d}-{date_begin.hour:02d}h{date_begin.minute:02d}_{date_begin.second:02d}s{_parameters.name_save}/' if not EPN else \
                   f'{saving_repo}'
    
    res_grid_search=""
    if (len(gsearch)>1):
        res_grid_search = "\n    The mean value of the maximum of the ROC (on the validation set) over all the loops is: "
        for i in range(len(gsearch)):
            res_grid_search += "\n      " + str(gsearch[i]) + f": \n        {mean_max_roc_auc_val[i]:.4f}+/-{std_max_roc_auc_val[i]:.4f}"
        res_grid_search += "\n    Thus, the best combination of hyperparameters for this model is: "+str(gsearch[np.argmax(mean_max_roc_auc_val)]) #if (_parameters.loss_fct not in _parameters.survival_list)\
            #else "\n    Thus, the best combination of hyperparameters for this model is: "+str(gsearch[np.argmin(mean_max_roc_auc_val)])
            # We have chosen to consider to be the best set of hyperparameters the set that has the highest value of the mean over each loop 
            # of the maximum value of the ROC AUC (on the epochs of the validation set).
            # To be clearer, we retrieve, over each loop, the maximum value of the ROC AUC on the validation set over all the epochs 
            # (for each combination of hyperparameters) and then we average these maximum values over each loop. We then choose the 
            # best combination of hyperparameters as the one having the maximum value once this averaging has been done. 
            # We could have used another way of choosing the best set of hyperparameters (such as choosing the hyperparameters that
            # are selected the most frequently over the different loops, or choosing the hyperparameters that have the highest maximum
            # value of the mean over the different loops of the ROC AUC of validation).
        
        hyperparams_names = sorted(gsearch[0].keys())
        configs = [[gsearch[i][hyperparams_names[j]] for i in range(len(gsearch))] for j in range(len(hyperparams_names))]
        columns = list(hyperparams_names)
        columns.append("AUROC_val")
        columns.append("std_AUROC_val")
        df_gsearch = pandas.DataFrame(data=np.transpose(np.vstack((configs, mean_max_roc_auc_val, std_max_roc_auc_val))), columns=columns)
        df_gsearch.to_csv(f"{repo_to_save}gsearch_results.csv")
           
    output = "\nTest results for cross-attention model:\n    Considered parameters:"
    for parameter_name, parameter_value in parameters_list.items():
        output += "\n      "+parameter_name+f": {parameter_value}"  
    output+="\n"

    if (len(gsearch)>1):  
        output+=res_grid_search+"\n"
    else:
        nb_test_splits = 5 if(_parameters.n_loops) else 1
        if (_parameters.loss_fct not in _parameters.survival_list):
            for test_s in range(nb_test_splits):
                for l in range(_parameters.n_loops):
                    output += f"\n-Test result test loop {test_s+1:01d}, loop {l+1:02d}: AU-ROC={metrics_test[test_s, l,1]:.3f}, Bal. acc={metrics_test[test_s, l,5]:.4f}, Acc={metrics_test[test_s, l,2]:.4f}, Sens={metrics_test[test_s, l,3]:.3f}, Spec={metrics_test[test_s, l,4]:.3f}, loss={metrics_test[test_s, l,0]:.2f}"
                    output += "\n "+str(confusion_mat[test_s, l])
                output += f"\nTest loop {test_s+1:01d} result: AU-ROC={np.mean(metrics_test[test_s, :,1]):.4f}+/-{np.std(metrics_test[test_s,:,1]):.4f}, Bal. acc={np.mean(metrics_test[test_s, :,5]):.4f}+/-{np.std(metrics_test[test_s, :,5]):.4f}, Acc={np.mean(metrics_test[test_s, :,2]):.4f}+/-{np.std(metrics_test[test_s, :,2]):.4f}, Sens={np.mean(metrics_test[test_s, :,3]):.3f}+/-{np.std(metrics_test[test_s, :,3]):.3f}, Spec={np.mean(metrics_test[test_s, :,4]):.3f}+/-{np.std(metrics_test[test_s, :,4]):.3f}, loss={np.mean(metrics_test[test_s, :,0]):.2f}+/-{np.std(metrics_test[test_s, :,0]):.2f}"
                if(max(parameters_list["nb_epochs"])!=0):
                    output += f'\nMean max validation AU-ROC: {np.mean(np.amax(np.mean(metrics_val[test_s, :, :, :, 1], axis=1), axis=1), axis=0):.4f}+/-{np.std(np.amax(np.mean(metrics_val[test_s, :, :, :, 1], axis=1), axis=1)):.4f}'
                    output += f'\nMax mean validation AU-ROC: {np.max(np.mean(np.mean(metrics_val[test_s, :, :, :, 1], axis=1), axis=0)):.4f}\n'
            output += f"\n\n => Global result: AU-ROC={np.mean(metrics_test[:,:,1]):.4f}+/-{np.std(metrics_test[:,:,1]):.4f}, Bal. acc={np.mean(metrics_test[:,:,5]):.4f}+/-{np.std(metrics_test[:,:,5]):.4f}, Acc={np.mean(metrics_test[:,:,2]):.4f}+/-{np.std(metrics_test[:,:,2]):.4f}, Sens={np.mean(metrics_test[:,:,3]):.3f}+/-{np.std(metrics_test[:,:,3]):.3f}, Spec={np.mean(metrics_test[:,:,4]):.3f}+/-{np.std(metrics_test[:,:,4]):.3f}, loss={np.mean(metrics_test[:,:,0]):.2f}+/-{np.std(metrics_test[:,:,0]):.2f}"
            if(max(parameters_list["nb_epochs"])!=0): output += f'\nMean max validation AU-ROC: {np.mean(np.amax(np.mean(metrics_val[:, :, :, :, 1], axis=2), axis=2), axis=(0,1)):.4f}+/-{np.std(np.amax(np.mean(metrics_val[:, :, :, :, 1], axis=2), axis=2)):.4f}'
            # output+= f'\nP-values: \n   - p-value test:    {np.mean(pvalues[:,:,:,0])}+/-{np.std(pvalues[:,:,:,0])}\n   - p-value all pat: {np.mean(pvalues[:,:,:,2])}+/-{np.std(pvalues[:,:,:,2])}'
        else:
            for test_s in range(nb_test_splits):
                for l in range(_parameters.n_loops):
                    output += f"\n-Test result test loop {test_s+1:01d}, loop {l+1:02d}: IBS={metrics_test[test_s, l,2]:.4f}, c-index={metrics_test[test_s, l,1]:.4f}, loss={metrics_test[test_s, l,0]:.3f}"
                output += f"\nTest loop {test_s+1:01d} result: IBS={np.mean(metrics_test[test_s, :,2]):.4f}+/-{np.std(metrics_test[test_s,:,2]):.4f}, c-index={np.mean(metrics_test[test_s, :,1]):.4f}+/-{np.std(metrics_test[test_s, :,1]):.4f}, loss={np.mean(metrics_test[test_s, :,0]):.3f}+/-{np.std(metrics_test[test_s, :,0]):.3f}"
                if(max(parameters_list["nb_epochs"])!=0):
                    output += f'\nMean min validation IBS: {np.mean(np.amin(np.mean(metrics_val[test_s, :, :, :, 2], axis=1), axis=1), axis=0):.4f}+/-{np.std(np.amin(np.mean(metrics_val[test_s, :, :, :, 2], axis=1), axis=1)):.4f}'
                    output += f'\nMin mean validation IBS: {np.min(np.mean(np.mean(metrics_val[test_s, :, :, :, 2], axis=1), axis=0)):.4f}\n'
            output += f"\n\n => Global result: IBS={np.mean(metrics_test[:,:,2]):.4f}+/-{np.std(metrics_test[:,:,2]):.4f}, c-index={np.mean(metrics_test[:,:,1]):.4f}+/-{np.std(metrics_test[:,:,1]):.4f}, loss={np.mean(metrics_test[:,:,0]):.3f}+/-{np.std(metrics_test[:,:,0]):.3f}"
            if(max(parameters_list["nb_epochs"])!=0): output += f'\nMean min validation IBS: {np.mean(np.amin(np.mean(metrics_val[:, :, :, :, 2], axis=2), axis=2), axis=(0,1)):.4f}+/-{np.std(np.amin(np.mean(metrics_val[:, :, :, :, 2], axis=2), axis=2)):.4f}'
            output+= f'\nP-values: \n   - p-value test - event time:    {np.mean(pvalues[:,:,:,0])}+/-{np.std(pvalues[:,:,:,0])}\n   - p-value all pat - event time: {np.mean(pvalues[:,:,:,2])}+/-{np.std(pvalues[:,:,:,2])}\n   - p-value test - AUSC:          {np.mean(pvalues[:,:,:,1])}+/-{np.std(pvalues[:,:,:,1])}\n   - p-value all pat - AUSC:       {np.mean(pvalues[:,:,:,3])}+/-{np.std(pvalues[:,:,:,3])}'

    print(output)


    if(save):
        output_file = open(f"{repo_to_save}results.txt", 'a')
        output_file.write(output)
        output_file.close()

        writer.add_text(f'global_results', output)
        writer.flush()

        date_end = datetime.datetime.today()
        writer.add_text(f'time/beginning_time', f'{date_begin.year}_{date_begin.month:02d}_{date_begin.day:02d}-{date_begin.hour:02d}h{date_begin.minute:02d}_{date_begin.second:02d}s')
        writer.add_text(f'time/end_time', f'{date_end.year}_{date_end.month:02d}_{date_end.day:02d}-{date_end.hour:02d}h{date_end.minute:02d}_{date_end.second:02d}s')
        writer.add_text(f'time/runtime', str((date_end-date_begin)))

        for k in range(len(gsearch)):
            for i in range(max(parameters_list["nb_epochs"])):
                for m in range(len(name_metrics)):
                    if(max(parameters_list["nb_epochs"])!=0):
                        writer.add_scalars(f'{name_metrics[m]}-validation-Gsearch/{list(gsearch[k].items())}', {f'test_loop_{j+1}':np.mean(metrics_val[j, :, k, i, m]) for j in range(len(metrics_test))}, i)
                        writer.add_scalar(f'{name_metrics[m]}-validation-Gsearch/{list(gsearch[k].items())}/mean_value',np.mean(metrics_val[:, :, k, i, m],axis=(0,1)), i)
                    # writer.add_scalar(f'{name_metrics[m]}-validation-Gsearch/{list(gsearch[k].items())}/std',np.std(metrics_val[:, :, k, i, m],axis=(0,1)), i)
                    if (len(gsearch)==1):  
                        writer.add_scalars(f'test/{name_metrics[m]}/', {f'test_loop_{j+1}':np.mean(metrics_test[j, :, m]) for j in range(len(metrics_test))})
                        writer.add_scalar(f'test/{name_metrics[m]}/mean_value', np.mean(metrics_test[:, m], axis=(0,1)))
                        writer.add_scalar(f'test/{name_metrics[m]}/std', np.std(metrics_test[:, :, m]))
    
        writer.flush()
    if (len(gsearch)==1 and max(parameters_list["nb_epochs"])!=0):  
        best_ep, metrics_best_epoch = select_best_epoch2(metrics_val, name_metrics)
        print(metrics_best_epoch)        
        if(max(parameters_list["nb_epochs"])>=10): plot_training_curves2(metrics_train, metrics_val, name_metrics, best_ep, writer, save)
    return 0