import sys 
import torch
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.utils import index_to_mask
from sklearn.metrics import balanced_accuracy_score, roc_curve, auc, f1_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from pycox.evaluation import EvalSurv
from torch.utils.data import Subset
###
sys.path.insert(1, './') 
from functions.models import MLP, MILModel
from functions.miscellaneous import pad_col
from functions.Scheduler import LRScheduler
from functions.MilDataset import collate
from functions.survival import visual_analysis_best_model, visual_analysis_best_model_our_data
m = nn.Sigmoid()

###
#Needed for train/test_surv:
def _pair_rank_mat(mat, idx_durations, events, dtype='float32'):
    n = len(idx_durations)
    for i in range(n):
        dur_i = idx_durations[i]
        ev_i = events[i]
        if ev_i == 0:
            continue
        for j in range(n):
            dur_j = idx_durations[j]
            ev_j = events[j]
            if (dur_i < dur_j) or ((dur_i == dur_j) and (ev_j == 0)):
                mat[i, j] = 1
    return mat
def pair_rank_mat(idx_durations, events, dtype='float32'):
    """Indicator matrix R with R_ij = 1{T_i < T_j and D_i = 1}.
    So it takes value 1 if we observe that i has an event before j and zero otherwise.
    
    Arguments:
        idx_durations {np.array} -- Array with durations.
        events {np.array} -- Array with event indicators.
    
    Keyword Arguments:
        dtype {str} -- dtype of array (default: {'float32'})
    
    Returns:
        np.array -- n x n matrix indicating if i has an observerd event before j.
    """
    idx_durations = idx_durations.reshape(-1)
    events = events.reshape(-1)
    n = len(idx_durations)
    mat = np.zeros((n, n), dtype=dtype)
    mat = _pair_rank_mat(mat, idx_durations, events, dtype)
    return mat


def train_EPN(dataset_to_send, train_loader, model, optimizer, criterion, m, lr_scheduler, device):
    dataset_to_send = dataset_to_send.reset_index()
    dict_UMAP = {}
    running_train_loss = 0.0
    correct = 0
    target_final, pred_final, prob_final = [], [], []
    # Training Loop
    model.train()
    torch.manual_seed(42) #To be used  ###########################
    for data in train_loader:
        #print(data)
        optimizer.zero_grad()  # zero the parameter gradients

        test_idx, labels = [],[]
        for i in range(len(data)):
            labels.append(data[i]['label'])
            test_idx.append(dataset_to_send[dataset_to_send['pat_id']==data[i]['pat_id']].index[0])

        predicted_outputs = model(dataset_to_send, training=True, test_idx=test_idx)  # predict output from the model
        clamped_preds = torch.minimum(torch.maximum(predicted_outputs, torch.zeros(len(predicted_outputs))),torch.ones(len(predicted_outputs)))
        # print(predicted_outputs)
        # print(clamped_preds)
        #dict_UMAP[data[i]['pat_id']] = {'in_MLP':in_MLP, 'label':data[i]['graphs'].y} 
                    
        target = torch.tensor(labels, device=device).float()            #data.y.unsqueeze(1).float()
        train_loss = criterion(clamped_preds, target)            #predicted_outputs, target)
        train_loss.backward()  # backpropagate the loss

        mean = torch.tensor([-10,-10,-10,-10], dtype=float)
        i=0
        # print('loss  '+str(train_loss.item()))
        # print(model._attn_cache)
        # print('weights')
        # for _, w in model.named_parameters():
        #     # print(w)
            # print(torch.mean(torch.abs(w)).item())
            # if(i==0 or i==2):
            #     fig, ax = plt.subplots()
            #     ax.stem(w.detach().numpy()[0, :200])
            #     # ax.set(xlim=(0, 8), xticks=np.arange(1, 8), ylim=(0, 8), yticks=np.arange(1, 8))
            #     plt.show()

        #     mean[i] = torch.mean(torch.abs(w.grad))
        #     i+=1
        # print('grad:    '+str(torch.mean(mean).item()))
        # print()
        
        

        # print(model.a)
        optimizer.step()  # adjust parameters based on the calculated gradients
        
        running_train_loss += train_loss.item()  # track the loss value  #to be divided by the batch size? (nope, taken care inside the function)

        pred = (clamped_preds >= 0.5).float()        #(m(predicted_outputs) >= 0.5).float()
        correct += int((pred == target).sum())  # Check against ground-truth labels.

        target_final.extend(target.tolist())
        pred_final.extend(pred.tolist())
        prob_final.extend(clamped_preds.tolist())        #(m(predicted_outputs).tolist())

    pred_final = np.array(pred_final)
    target_final = np.array(target_final)
    prob_final = np.array(prob_final)

    #Get accuracies
    train_accuracy = correct / len(train_loader.dataset)
    balanced_accuracy = balanced_accuracy_score(target_final, pred_final)

    # Calculate training loss value
    train_loss_value = running_train_loss / len(train_loader)
    # print(train_loss_value)
    lr_scheduler(train_loss_value)

    #calculate AUC
    fpr, tpr, threshold = roc_curve(target_final, prob_final)
    roc_auc = auc(fpr, tpr)
    #print('ROC-AUC value ', roc_auc)
    prec, recall, _, _ = precision_recall_fscore_support(target_final, pred_final, average='binary', zero_division=0)

    cm = confusion_matrix(target_final, pred_final)
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]

    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))

    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))

    #calculate F1-scores
    #print('Micro F1 score ', f1_score(y_true=target_final, y_pred=pred_final, average='micro'))
    #print('Macro F1 score ', f1_score(y_true=target_final, y_pred=target_final, average='macro'))
    #print('Weighted F1 score ', f1_score(y_true=target_final, y_pred=target_final, average='weighted'))

    return roc_auc, train_loss_value, conf_sensitivity, conf_specificity, train_accuracy, balanced_accuracy, prec, recall, prob_final, target_final, dict_UMAP




def test_EPN(dataset_to_send, test_loader, model, m, criterion, test_pat_ids, device):
    dataset_to_send = dataset_to_send.reset_index()
    # dict_UMAP, dict_out_catt = {},{}
    pred, lab, prob = [], [], []
    running_test_loss = 0.0 
    model.eval() #to set dropout and batch normalisation layers to evaluation mode
    with torch.no_grad():
        correct = 0
        torch.manual_seed(42) #To be used #############################
        for i, data in enumerate(test_loader):
            id_pat_pb=-1
            test_idx, labels, full_test_idx = [],[],[]
            for i in range(len(data)):
                labels.append(data[i]['label'])
                test_idx.append(dataset_to_send[dataset_to_send['pat_id']==data[i]['pat_id']].index[0])
                # if(data[i]['pat_id']==11011101241006):id_pat_pb=dataset_to_send[dataset_to_send['pat_id']==data[i]['pat_id']].index[0]
                # print(data[i]['pat_id'])
            
            for test_pat_id in test_pat_ids:
                full_test_idx.append(dataset_to_send[dataset_to_send['pat_id']==test_pat_id].index[0])

            predicted_outputs = model(dataset_to_send, training=False, test_idx=test_idx, full_test_idx=full_test_idx, id_pat_pb=id_pat_pb)  # predict output from the model
            # clamped_preds = predicted_outputs
            clamped_preds = torch.minimum(torch.maximum(predicted_outputs, torch.zeros(len(predicted_outputs))),torch.ones(len(predicted_outputs)))
            # print(predicted_outputs)
            # print(clamped_preds)
            #dict_UMAP[data[i]['pat_id']] = {'in_MLP':in_MLP, 'label':data[i]['graphs'].y} 
                    
            target = torch.tensor(labels, device=device).float() #data.y.unsqueeze(1).float()
            prediction = (clamped_preds >= 0.5).float()              #(m(predicted_outputs) >= 0.5).float()
            correct += int((prediction == target).sum())  # Check against ground-truth labels.

            batch_prob = clamped_preds.tolist()                 #m(predicted_outputs).tolist()
            batch_pred = prediction.tolist()
            batch_lab = labels

            lab.extend(batch_lab)
            pred.extend(batch_pred)
            prob.extend(batch_prob)

            test_loss = criterion(clamped_preds, target)                   #(predicted_outputs, target)
            running_test_loss += test_loss.item()  # track the loss value

    test_accuracy = correct / len(test_loader.dataset)
    test_loss_value = running_test_loss / len(test_loader)
    pred = torch.tensor(pred)
    lab = torch.tensor(lab)
    prob = torch.tensor(prob)

    # print('Classification')
    # print('Micro F1 score ', f1_score(y_true=lab, y_pred=pred, average='micro'))
    # print('Macro F1 score ', f1_score(y_true=lab, y_pred=pred, average='macro'))
    # print('Weighted F1 score ', f1_score(y_true=lab, y_pred=pred, average='weighted'))

    fpr, tpr, threshold = roc_curve(lab, prob)
    roc_auc = auc(fpr, tpr)
    prec, recall, _, _ = precision_recall_fscore_support(lab, pred, average='binary', zero_division=0)

    cm = confusion_matrix(lab, pred, labels=[0,1])

    # print(cm)

    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]

    # calculate the accuracy
    accuracy = ((TP + TN) / float(TP + TN + FP + FN))

    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))

    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))

    # balanced accuracy
    bal_accuracy = (conf_specificity+conf_sensitivity) / 2
    
    return  lab, roc_auc, tpr, prob, conf_sensitivity, conf_specificity, accuracy, bal_accuracy, test_loss_value, prec, recall, cm, prob, lab



def train_EPN_surv(dataset_to_send, train_loader, model, optimizer, criterion, m, lr_scheduler, parameters, device):
    # print(dataset_to_send['pat_id'])
    dataset_to_send = dataset_to_send.reset_index()
    # print(dataset_to_send['pat_id'])
    dict_UMAP = {}
    running_train_loss = 0.0
    GAINED_db = len(np.where(['feat_' in dataset_to_send.columns[i] for i in range(len(dataset_to_send.columns))])[0])==0
    # print('GAINED_db', GAINED_db)
    cols_feat_all = [dataset_to_send.columns[i] for i in range(len(dataset_to_send.columns)) if 'feat_' in dataset_to_send.columns[i]] if(not GAINED_db) else \
                    [f'{parameters.feats_usable_surv[feat_nb]}_{i+1}' for feat_nb in range(len(parameters.feats_usable_surv)) for i in range(parameters.feats_usable_surv_dim[feat_nb])]
    cols_feat = cols_feat_all if(not GAINED_db) else [f'{parameters.feats_usable_surv[feat_nb]}_{i+1}' for feat_nb in range(len(parameters.feats_usable_surv)) for i in range(parameters.feats_usable_surv_dim[feat_nb]) if(parameters.feats_usable_surv[feat_nb] in parameters.feats_for_popg)]
    cols_pred_float = sorted(dataset_to_send.columns.difference(cols_feat_all+['label_duration', 'label_event', 'split', 'pat_id', 'index', 'pat_to_compute']).values.astype(float)) #'index' comes from the reset_index() operation
    # Training Loop
    model.train()
    torch.manual_seed(42) #To be used  ###########################
    for data in train_loader:
        optimizer.zero_grad()  # zero the parameter gradients

        dataset_to_send['batch'] = False
        for i in range(len(data)):
            dataset_to_send.loc[dataset_to_send['pat_id']==data[i]['pat_id'],'batch'] = True

        predicted_outputs = model(dataset_to_send)#, training=True, test_idx=test_idx)  # predict output from the model #torch matrix (one vector by patient))
        # print('b3')
        # print(datetime.datetime.today())
        clamped_preds = torch.clamp(predicted_outputs, 0, 1) #minimum(torch.maximum(predicted_outputs, torch.zeros_like(predicted_outputs)),torch.ones_like(predicted_outputs))
        # print(predicted_outputs)
        # print(clamped_preds)
        #dict_UMAP[data[i]['pat_id']] = {'in_MLP':in_MLP, 'label':data[i]['graphs'].y} 
                    
        target = torch.tensor(dataset_to_send.loc[dataset_to_send['batch'],('label_duration', 'label_event')].values, device=device).float()   #data.y.unsqueeze(1).float()
        timepoints_pred = cols_pred_float
        if(not GAINED_db):
            visual_analysis_best_model(timepoints_pred, dataset_to_send.loc[dataset_to_send['batch'],[str(cols_pred_float[i]) for i in range(len(cols_pred_float))]].values, clamped_preds, target)
        else:
            visual_analysis_best_model_our_data(timepoints_pred, dataset_to_send.loc[dataset_to_send['batch'],[str(cols_pred_float[i]) for i in range(len(cols_pred_float))]].values, clamped_preds, target)
        print(truc.a)
        train_loss = criterion(timepoints_pred, clamped_preds, target, GAINED_db) #use the batch info  #target and timepoints_pred have no grad, but clamped_preds has
        train_loss.backward()  # backpropagate the loss
        # print('grad: to see if computed')
        # for a, w in model.named_parameters():
        #     print(a)
        #     print(torch.mean(torch.abs(w.grad)))
            
        # print(model.a)
        optimizer.step()  # adjust parameters based on the calculated gradients
        
        running_train_loss += train_loss.item()  # track the loss value  #to be divided by the batch size? (nope, taken care inside the function)

        try:
            target_final = pd.concat([target_final, dataset_to_send.loc[dataset_to_send['batch'],('label_duration', 'label_event')]], ignore_index=True)
            prob_final = pd.concat([prob_final, pd.DataFrame(clamped_preds.detach(), columns=cols_pred_float)], ignore_index=True)
        except NameError:
            target_final = dataset_to_send.loc[dataset_to_send['batch'],('label_duration', 'label_event')]
            prob_final = pd.DataFrame(clamped_preds.detach(), columns=cols_pred_float)
    #     print(prob_final)
    # print(target_final)
    # print(prob_final)

    # Compute training loss value
    train_loss_value = running_train_loss / len(train_loader)
    print(train_loss_value)
    lr_scheduler(train_loss_value)

    # Compute c-index and IBS
    ev = EvalSurv(prob_final.transpose(), target_final['label_duration'].values, target_final['label_event'].values, censor_surv='km')
    time_grid = np.linspace(target_final['label_duration'].values.min(), target_final['label_duration'].values.max(), 100)
    c_index = ev.concordance_td()
    IBS = ev.integrated_brier_score(time_grid)  #loss, c-index, IBS #no loss computation for the moment

    return train_loss_value, c_index, IBS, prob_final, target_final, dict_UMAP




def test_EPN_surv(dataset_to_send, test_loader, model, m, criterion, test_pat_ids, parameters, device):
    dataset_to_send = dataset_to_send.reset_index()
    # dict_UMAP, dict_out_catt = {},{}
    running_test_loss = 0.0 
    GAINED_db = len(np.where(['feat_' in dataset_to_send.columns[i] for i in range(len(dataset_to_send.columns))])[0])==0
    # print('GAINED_db', GAINED_db)
    cols_feat_all = [dataset_to_send.columns[i] for i in range(len(dataset_to_send.columns)) if 'feat_' in dataset_to_send.columns[i]] if(not GAINED_db) else \
                    [f'{parameters.feats_usable_surv[feat_nb]}_{i+1}' for feat_nb in range(len(parameters.feats_usable_surv)) for i in range(parameters.feats_usable_surv_dim[feat_nb])]
    cols_feat = cols_feat_all if(not GAINED_db) else [f'{parameters.feats_usable_surv[feat_nb]}_{i+1}' for feat_nb in range(len(parameters.feats_usable_surv)) for i in range(parameters.feats_usable_surv_dim[feat_nb]) if(parameters.feats_usable_surv[feat_nb] in parameters.feats_for_popg)]
    cols_pred_float = sorted(dataset_to_send.columns.difference(cols_feat_all+['label_duration', 'label_event', 'split', 'pat_id', 'index', 'pat_to_compute']).values.astype(float)) #'index' comes from the reset_index() operation
    cols_pred_str = [str(cols_pred_float[i]) for i in range(len(cols_pred_float))]
    model.eval() #to set dropout and batch normalisation layers to evaluation mode
    with torch.no_grad():
        torch.manual_seed(42) #To be used #############################
        for i, data in enumerate(test_loader):
            dataset_to_send['batch'] = False
            for i in range(len(data)):
                dataset_to_send.loc[dataset_to_send['pat_id']==data[i]['pat_id'],'batch'] = True
            # print('dataset_to_send (train_test_fcts): check the "batch" use')
            # print(dataset_to_send)
            pred_MLP = dataset_to_send[dataset_to_send['batch']][cols_pred_str]

            predicted_outputs = model(dataset_to_send) #, training=False, test_idx=test_idx, full_test_idx=full_test_idx, id_pat_pb=id_pat_pb)  # predict output from the model
            clamped_preds = torch.clamp(predicted_outputs, 0, 1)#minimum(torch.maximum(predicted_outputs, torch.zeros_like(predicted_outputs)),torch.ones_like(predicted_outputs))
            # print(predicted_outputs)
            # print(clamped_preds)
            #dict_UMAP[data[i]['pat_id']] = {'in_MLP':in_MLP, 'label':data[i]['graphs'].y} 
                    
            # target = torch.tensor(dataset_to_send.loc[dataset_to_send['batch'],('label_duration', 'label_event')].values, device=device).float()   #data.y.unsqueeze(1).float()
            # timepoints_pred = cols_pred_float
            # visual_analysis_best_model(timepoints_pred, dataset_to_send.loc[dataset_to_send['batch'],[str(cols_pred_float[i]) for i in range(len(cols_pred_float))]].values, clamped_preds, target)
            # print(truc.a)
            # test_loss = criterion(timepoints_pred, clamped_preds, target_final) #use batch
            # running_test_loss += test_loss.item()  # track the loss value

            try:
                target_final = pd.concat([target_final, dataset_to_send.loc[dataset_to_send['batch'],('label_duration', 'label_event')]], ignore_index=True)
                prob_final = pd.concat([prob_final, pd.DataFrame(clamped_preds.detach(), columns=cols_pred_float)], ignore_index=True)
                prob_MLP = pd.concat([prob_MLP, pred_MLP.rename(columns={cols_pred_str[i]:cols_pred_float[i] for i in range(len(cols_pred_float))})], ignore_index=True)
            except NameError:
                target_final = dataset_to_send.loc[dataset_to_send['batch'],('label_duration', 'label_event')]
                prob_final = pd.DataFrame(clamped_preds.detach(), columns=cols_pred_float)
                prob_MLP = pred_MLP.rename(columns={cols_pred_str[i]:cols_pred_float[i] for i in range(len(cols_pred_float))})
        print(prob_MLP)

    # print(target_final)
    # print(prob_final)

    # Compute loss, c-index and IBS
    # test_loss_value = running_test_loss / len(test_loader)
    ev = EvalSurv(prob_final.transpose(), target_final['label_duration'].values, target_final['label_event'].values, censor_surv='km')
    time_grid = np.linspace(target_final['label_duration'].values.min(), target_final['label_duration'].values.max(), 100)
    c_index = ev.concordance_td()
    IBS = ev.integrated_brier_score(time_grid)  #loss, c-index, IBS #no loss computation for the moment
    


    metrics_bonus_test = {}
    # Calculate metrics only for uncensored patients:
    mask_uncensored_results = torch.tensor(target_final['label_event'].values==1)
    target_duration_cont_final_uncensored = target_final['label_duration'].values[mask_uncensored_results]
    target_event_final_uncensored = target_final['label_event'].values[mask_uncensored_results]
    time_grid_uncensored = np.linspace(np.min(target_duration_cont_final_uncensored), np.max(target_duration_cont_final_uncensored), 100) #LEN_BS=100
    ev = EvalSurv(prob_final.transpose().iloc[:, np.where(mask_uncensored_results)[0]], target_duration_cont_final_uncensored, target_event_final_uncensored, censor_surv='km')
    metrics_bonus_test['cindex_uncensored'] = ev.concordance_td('antolini')
    metrics_bonus_test['bs_uncensored'] = ev.brier_score(time_grid_uncensored)
    metrics_bonus_test['ibs_uncensored'] = ev.integrated_brier_score(time_grid_uncensored) 

    ev = EvalSurv(prob_MLP.transpose().iloc[:, np.where(mask_uncensored_results)[0]], target_duration_cont_final_uncensored, target_event_final_uncensored, censor_surv='km')
    metrics_bonus_test['MLP_cindex_uncensored'] = ev.concordance_td('antolini')
    metrics_bonus_test['MLP_bs_uncensored'] = ev.brier_score(time_grid_uncensored)
    metrics_bonus_test['MLP_ibs_uncensored'] = ev.integrated_brier_score(time_grid_uncensored) 

    # Calculate metrics only for censored patients:
    mask_censored_results = torch.tensor(target_final['label_event'].values==0)
    target_duration_cont_final_censored = target_final['label_duration'].values[mask_censored_results]
    target_event_final_censored = target_final['label_event'].values[mask_censored_results]
    time_grid_censored = np.linspace(np.min(target_duration_cont_final_censored), np.max(target_duration_cont_final_censored), 100) #LEN_BS=100
    ev = EvalSurv(prob_final.transpose().iloc[:, np.where(mask_censored_results)[0]], target_duration_cont_final_censored, target_event_final_censored, censor_surv='km')
    metrics_bonus_test['bs_censored'] = ev.brier_score(time_grid_censored)
    metrics_bonus_test['ibs_censored'] = ev.integrated_brier_score(time_grid_censored) 

    ev = EvalSurv(prob_MLP.transpose().iloc[:, np.where(mask_censored_results)[0]], target_duration_cont_final_censored, target_event_final_censored, censor_surv='km')
    metrics_bonus_test['MLP_bs_censored'] = ev.brier_score(time_grid_censored)
    metrics_bonus_test['MLP_ibs_censored'] = ev.integrated_brier_score(time_grid_censored) 
    print(np.count_nonzero(mask_censored_results | mask_uncensored_results), np.count_nonzero(mask_censored_results & mask_uncensored_results))
    print(np.count_nonzero(mask_censored_results | mask_uncensored_results), np.count_nonzero(mask_censored_results))



    ratio_early = 110/190
    ratio_late = 80/190
    # Calculate metrics only for early patients:
    mask_late_results = index_to_mask(torch.topk(torch.tensor(target_final['label_duration'].values), k=int(ratio_early*len(target_final['label_duration'])))[1], size=len(target_final['label_duration']))
    mask_early_results = ~mask_late_results

    target_duration_cont_final_early = target_final['label_duration'].values[mask_early_results]
    target_event_final_early = target_final['label_event'].values[mask_early_results]
    time_grid_early = np.linspace(np.min(target_duration_cont_final_early), np.max(target_duration_cont_final_early), 100) #LEN_BS=100
    ev = EvalSurv(prob_final.transpose().iloc[:, np.where(mask_early_results)[0]], target_duration_cont_final_early, target_event_final_early, censor_surv='km')
    metrics_bonus_test['cindex_early'] = ev.concordance_td('antolini')
    metrics_bonus_test['bs_early'] = ev.brier_score(time_grid_early)
    metrics_bonus_test['ibs_early'] = ev.integrated_brier_score(time_grid_early) 

    ev = EvalSurv(prob_MLP.transpose().iloc[:, np.where(mask_early_results)[0]], target_duration_cont_final_early, target_event_final_early, censor_surv='km')
    metrics_bonus_test['MLP_cindex_early'] = ev.concordance_td('antolini')
    metrics_bonus_test['MLP_bs_early'] = ev.brier_score(time_grid_early)
    metrics_bonus_test['MLP_ibs_early'] = ev.integrated_brier_score(time_grid_early) 

    # Calculate metrics only for late patients:
    target_duration_cont_final_late = target_final['label_duration'].values[mask_late_results]
    target_event_final_late = target_final['label_event'].values[mask_late_results]
    time_grid_late = np.linspace(np.min(target_duration_cont_final_late), np.max(target_duration_cont_final_late), 100) #LEN_BS=100
    ev = EvalSurv(prob_final.transpose().iloc[:, np.where(mask_late_results)[0]], target_duration_cont_final_late, target_event_final_late, censor_surv='km')
    metrics_bonus_test['cindex_late'] = ev.concordance_td('antolini')
    metrics_bonus_test['bs_late'] = ev.brier_score(time_grid_late)
    metrics_bonus_test['ibs_late'] = ev.integrated_brier_score(time_grid_late) 

    ev = EvalSurv(prob_MLP.transpose().iloc[:, np.where(mask_late_results)[0]], target_duration_cont_final_late, target_event_final_late, censor_surv='km')
    metrics_bonus_test['MLP_cindex_late'] = ev.concordance_td('antolini')
    metrics_bonus_test['MLP_bs_late'] = ev.brier_score(time_grid_late)
    metrics_bonus_test['MLP_ibs_late'] = ev.integrated_brier_score(time_grid_late) 

    return  c_index, IBS, prob_final, target_final, metrics_bonus_test #we can't compute loss value because we do not have the predictions for the time of event of val/test patients
 




def train_MLP_surv(train_loader, model1, optimizer, criterion, lr_scheduler, time_limits=None, time_grid=None):

    running_train_loss = 0.0
    target_duration_disc_final, target_duration_cont_final, target_event_final, pred_final = [], [], [], []
    # Training Loop
    model1.train()
    for data in train_loader:
        optimizer.zero_grad()  # zero the parameter gradients
        predicted_output = model1(data.x)  # predict output from the model
        target_duration_disc = torch.tensor([data.y[2*i] for i in range(len(data.x))])
        target_duration_cont = torch.tensor([data.g[i] for i in range(len(data.x))])
        target_event = torch.tensor([data.y[2*i+1] for i in range(len(data.x))], dtype=float)

        #Compute the loss:
        rank_mat = torch.tensor(pair_rank_mat(target_duration_disc, target_event))
        train_loss = criterion(predicted_output, target_duration_disc, target_event, rank_mat) 

        train_loss.backward()  # backpropagate the loss
        optimizer.step()  # adjust parameters based on the calculated gradients
        running_train_loss += train_loss.item()  # track the loss value

        target_duration_disc_final.extend(target_duration_disc.tolist())
        target_duration_cont_final.extend(target_duration_cont.tolist())
        target_event_final.extend(target_event.tolist())
        pred_final.extend(predicted_output.tolist())

    pred_final = np.array(pred_final)
    target_duration_disc_final = np.array(target_duration_disc_final)
    target_duration_cont_final = np.array(target_duration_cont_final)
    target_event_final = np.array(target_event_final)
    target_final = np.array([target_duration_disc_final, target_event_final, target_duration_cont_final])

    # Calculate training loss value
    train_loss_value = running_train_loss / len(train_loader)
    lr_scheduler(train_loss_value)

    # Calculate metrics:
    pmf = pad_col(torch.tensor(pred_final)).softmax(1)[:, :-1]
    surv = 1 - pmf.cumsum(1)
    ev = EvalSurv(pd.DataFrame(np.array(surv).T, time_limits), target_duration_cont_final, target_event_final, censor_surv='km')
    cindex = ev.concordance_td('antolini')
    ibs = ev.integrated_brier_score(time_grid) 


    # #If needed, plot Brier score at every epoch:
    # bs = ev.brier_score(time_grid)
    # fig = plt.figure() 
    # ax = fig.add_subplot(111)
    # ax.plot(time_grid, bs)
    # plt.title("Training Brier score")
    # plt.xlabel("Time")
    # plt.show()

    return train_loss_value, cindex, ibs, pred_final, target_final 



def test_MLP_surv(test_loader, model, criterion, time_limits=None, time_grid=None):
    target_duration_disc_final, target_duration_cont_final, target_event_final, pred_final = [], [], [], []
    running_test_loss = 0.0 # Ajout
    model.eval() #to set dropout and batch normalisation layers to evaluation mode
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            
            predicted_output = model(data.x)  # predict output from the model
            target_duration_disc = torch.tensor([data.y[2*i] for i in range(len(data.x))])
            target_duration_cont = torch.tensor([data.g[i] for i in range(len(data.x))])
            target_event = torch.tensor([data.y[2*i+1] for i in range(len(data.x))], dtype=float)

            #Compute the loss:
            rank_mat = torch.tensor(pair_rank_mat(target_duration_disc, target_event))
            test_loss = criterion(predicted_output, target_duration_disc, target_event, rank_mat) 

            running_test_loss += test_loss.item()  # track the loss value

            target_duration_disc_final.extend(target_duration_disc.tolist())
            target_duration_cont_final.extend(target_duration_cont.tolist())
            target_event_final.extend(target_event.tolist())
            pred_final.extend(predicted_output.tolist())

    pred_final = np.array(pred_final)
    target_duration_disc_final = np.array(target_duration_disc_final)
    target_duration_cont_final = np.array(target_duration_cont_final)
    target_event_final = np.array(target_event_final)
    target_final = np.array([target_duration_disc_final, target_event_final, target_duration_cont_final])
    test_loss_value = running_test_loss / len(test_loader)

    # Calculate metrics:
    pmf = pad_col(torch.tensor(pred_final)).softmax(1)[:, :-1]
    surv = 1 - pmf.cumsum(1)
    ev = EvalSurv(pd.DataFrame(np.array(surv).T, time_limits), target_duration_cont_final, target_event_final, censor_surv='km')
    cindex = ev.concordance_td('antolini')
    bs = ev.brier_score(time_grid)
    ibs = ev.integrated_brier_score(time_grid) 

    return test_loss_value, cindex, bs, ibs, pred_final, target_final











def train_LogReg(train_data, train_label, model1):
    model1.fit(train_data, train_label)
    proba = model1.predict_proba(train_data)
    pred = np.argmax(proba, axis=1).astype(float)
    correct = int((pred == train_label).sum())
    train_loss_value = None

    #Get accuraciesee'
    train_accuracy = correct / len(train_data)
    balanced_accuracy = balanced_accuracy_score(train_label, pred)

    #calculate AUC
    fpr, tpr, threshold = roc_curve(train_label, proba[:,1])
    roc_auc = auc(fpr, tpr)
    #print('ROC-AUC value ', roc_auc)

    cm = confusion_matrix(train_label, pred)
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]

    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))

    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))

    #calculate F1-scores
    #print('Micro F1 score ', f1_score(y_true=target_final, y_pred=pred_final, average='micro'))
    #print('Macro F1 score ', f1_score(y_true=target_final, y_pred=target_final, average='macro'))
    #print('Weighted F1 score ', f1_score(y_true=target_final, y_pred=target_final, average='weighted'))

    return balanced_accuracy, roc_auc, conf_sensitivity, conf_specificity, train_accuracy, train_loss_value, cm, proba[:,1], train_label


def train_MLP(train_loader, model1, optimizer, criterion, m, lr_scheduler):

    running_train_loss = 0.0
    correct = 0
    target_final, pred_final, prob_final = [], [], []
    # Training Loop
    model1.train()
    for data in train_loader:
        optimizer.zero_grad()  # zero the parameter gradients
        predicted_output = model1(data.x)  # predict output from the model
        target = data.y.unsqueeze(1).float()

        train_loss = criterion(predicted_output, target)
        train_loss.backward()  # backpropagate the loss
        optimizer.step()  # adjust parameters based on the calculated gradients
        running_train_loss += train_loss.item()  # track the loss value

        pred = (m(predicted_output) >= 0.5).float()
        correct += int((pred == target).sum())  # Check against ground-truth labels.

        target_final.extend(target.tolist())
        pred_final.extend(pred.tolist())
        prob_final.extend(m(predicted_output).tolist())

    pred_final = np.array(pred_final)
    target_final = np.array(target_final)
    prob_final = np.array(prob_final)

    #Get accuracies
    train_accuracy = correct / len(train_loader.dataset)
    balanced_accuracy = balanced_accuracy_score(target_final, pred_final)

    # Calculate training loss value
    train_loss_value = running_train_loss / len(train_loader)
    lr_scheduler(train_loss_value)

    #calculate AUC
    fpr, tpr, threshold = roc_curve(target_final, prob_final)
    roc_auc = auc(fpr, tpr)
    #print('ROC-AUC value ', roc_auc)

    cm = confusion_matrix(target_final, pred_final)
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]

    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))

    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))

    #calculate F1-scores
    #print('Micro F1 score ', f1_score(y_true=target_final, y_pred=pred_final, average='micro'))
    #print('Macro F1 score ', f1_score(y_true=target_final, y_pred=target_final, average='macro'))
    #print('Weighted F1 score ', f1_score(y_true=target_final, y_pred=target_final, average='weighted'))

    return balanced_accuracy, roc_auc, conf_sensitivity, conf_specificity, train_accuracy, train_loss_value, cm, prob_final, target_final


def train_GNN(train_loader, model1, optimizer, criterion, m, lr_scheduler):

    running_train_loss = 0.0
    correct = 0
    target_final, pred_final, prob_final = [], [], []
    # Training Loop
    model1.train()
    for data in train_loader:
        optimizer.zero_grad()  # zero the parameter gradients
        predicted_output = model1(data.x, data.edge_index, data.edge_attr, data.batch)  # predict output from the model
        target = data.y.unsqueeze(1).float()

        train_loss = criterion(predicted_output, target)
        train_loss.backward()  # backpropagate the loss
        optimizer.step()  # adjust parameters based on the calculated gradients
        running_train_loss += train_loss.item()  # track the loss value

        pred = (m(predicted_output) >= 0.5).float()
        correct += int((pred == target).sum())  # Check against ground-truth labels.

        target_final.extend(target.tolist())
        pred_final.extend(pred.tolist())
        prob_final.extend(m(predicted_output).tolist())

    pred_final = np.array(pred_final)
    target_final = np.array(target_final)
    prob_final = np.array(prob_final)

    #Get accuracies
    train_accuracy = correct / len(train_loader.dataset)
    balanced_accuracy = balanced_accuracy_score(target_final, pred_final)

    # Calculate training loss value
    train_loss_value = running_train_loss / len(train_loader)
    lr_scheduler(train_loss_value)

    #calculate AUC
    fpr, tpr, threshold = roc_curve(target_final, prob_final)
    roc_auc = auc(fpr, tpr)
    #print('ROC-AUC value ', roc_auc)

    cm = confusion_matrix(target_final, pred_final)
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]

    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))

    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))

    #calculate F1-scores
    #print('Micro F1 score ', f1_score(y_true=target_final, y_pred=pred_final, average='micro'))
    #print('Macro F1 score ', f1_score(y_true=target_final, y_pred=target_final, average='macro'))
    #print('Weighted F1 score ', f1_score(y_true=target_final, y_pred=target_final, average='weighted'))

    return balanced_accuracy, roc_auc, conf_sensitivity, conf_specificity, train_accuracy, train_loss_value, cm, prob_final, target_final


def train_MIL(train_loader, model1, optimizer, criterion, m, lr_scheduler):

    running_train_loss = 0.0
    correct = 0
    target_final, pred_final, prob_final = [], [], []
    # Training Loop
    model1.train()
    for data, bagids, labels in train_loader:
        optimizer.zero_grad()  # zero the parameter gradients
        predicted_output = model1((data, bagids))  # predict output from the model
        target = labels.unsqueeze(1).float()

        train_loss = criterion(predicted_output, target)
        train_loss.backward()  # backpropagate the loss
        optimizer.step()  # adjust parameters based on the calculated gradients
        running_train_loss += train_loss.item()  # track the loss value

        pred = (m(predicted_output) >= 0.5).float()
        correct += int((pred == target).sum())  # Check against ground-truth labels.

        target_final.extend(target.tolist())
        pred_final.extend(pred.tolist())
        prob_final.extend(m(predicted_output).tolist())

    pred_final = np.array(pred_final)
    target_final = np.array(target_final)
    prob_final = np.array(prob_final)

    #Get accuracies
    train_accuracy = correct / len(train_loader.dataset)
    balanced_accuracy = balanced_accuracy_score(target_final, pred_final)

    # Calculate training loss value
    train_loss_value = running_train_loss / len(train_loader)
    lr_scheduler(train_loss_value)

    #calculate AUC
    fpr, tpr, threshold = roc_curve(target_final, prob_final)
    roc_auc = auc(fpr, tpr)
    #print('ROC-AUC value ', roc_auc)

    cm = confusion_matrix(target_final, pred_final)
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]

    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))

    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))

    #calculate F1-scores
    #print('Micro F1 score ', f1_score(y_true=target_final, y_pred=pred_final, average='micro'))
    #print('Macro F1 score ', f1_score(y_true=target_final, y_pred=target_final, average='macro'))
    #print('Weighted F1 score ', f1_score(y_true=target_final, y_pred=target_final, average='weighted'))

    return balanced_accuracy, roc_auc, conf_sensitivity, conf_specificity, train_accuracy, train_loss_value, cm, prob_final, target_final


def test_LogReg(data, model1):
    test_data = [np.array(list(zip(*data))[0][i][1])[0] for i in range(len(list(zip(*data))[0]))]
    labels = [np.array(list(zip(*data))[1][i][1]) for i in range(len(list(zip(*data))[0]))]

    proba = model1.predict_proba(test_data)
    pred = np.argmax(proba, axis=1).astype(float)
    correct = int((pred == labels).sum())
    test_loss_value = 0

    #Get accuraciesee'
    test_accuracy = correct / len(test_data)
    balanced_accuracy = balanced_accuracy_score(labels, pred)

    #calculate AUC
    fpr, tpr, threshold = roc_curve(labels, proba[:,1])
    roc_auc = auc(fpr, tpr)
    #print('ROC-AUC value ', roc_auc)

    cm = confusion_matrix(labels, pred)
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]

    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))

    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))

    #calculate F1-scores
    #print('Micro F1 score ', f1_score(y_true=target_final, y_pred=pred_final, average='micro'))
    #print('Macro F1 score ', f1_score(y_true=target_final, y_pred=target_final, average='macro'))
    #print('Weighted F1 score ', f1_score(y_true=target_final, y_pred=target_final, average='weighted'))

    return labels, roc_auc, tpr, proba[:,1], conf_sensitivity, conf_specificity, test_accuracy, balanced_accuracy, test_loss_value, cm, proba[:,1], labels
 


def test_MLP(test_loader, model1, m, criterion):
    pred, lab, prob = [], [], []
    running_test_loss = 0.0 # Ajout
    model1.eval() #to set dropout and batch normalisation layers to evaluation mode
    with torch.no_grad():
        correct = 0
        for i, data in enumerate(test_loader):
            predicted_output = model1(data.x)
            target = data.y.unsqueeze(1).float()

            prediction = (m(predicted_output) >= 0.5).float()
            correct += int((prediction == target).sum())  # Check against ground-truth labels.

            batch_prob = m(predicted_output).tolist()
            batch_pred = prediction.tolist()
            batch_lab = data.y.tolist()

            lab.extend(batch_lab)
            pred.extend(batch_pred)
            prob.extend(batch_prob)

            test_loss = criterion(predicted_output, target)
            running_test_loss += test_loss.item()  # track the loss value

    test_accuracy = correct / len(test_loader.dataset)
    test_loss_value = running_test_loss / len(test_loader)
    pred = np.array(pred)
    lab = np.array(lab)
    prob = np.array(prob)

    # print('Classification')
    # print('Micro F1 score ', f1_score(y_true=lab, y_pred=pred, average='micro'))
    # print('Macro F1 score ', f1_score(y_true=lab, y_pred=pred, average='macro'))
    # print('Weighted F1 score ', f1_score(y_true=lab, y_pred=pred, average='weighted'))

    fpr, tpr, threshold = roc_curve(lab, prob)
    roc_auc = auc(fpr, tpr)

    cm = confusion_matrix(lab, pred)

    # print(cm)

    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]

    # calculate the accuracy
    accuracy = ((TP + TN) / float(TP + TN + FP + FN))

    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))

    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))

    # balanced accuracy
    bal_accuracy = (conf_specificity+conf_sensitivity) / 2

    # print("bal acc", bal_accuracy, "AUC", roc_auc, "sens", conf_sensitivity, "spc", conf_specificity)

    # # Visualize
    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')

    # plt.xlabel('Predictions', fontsize=18)
    # plt.ylabel('Actuals', fontsize=18)
    # plt.title('Confusion Matrix', fontsize=18)
    # plt.show()

    return  lab, roc_auc, tpr, prob, conf_sensitivity, conf_specificity, accuracy, bal_accuracy, test_loss_value, cm, prob, lab



def test_GNN(test_loader, model1, m, criterion):
    pred, lab, prob = [], [], []
    running_test_loss = 0.0 
    model1.eval() #to set dropout and batch normalisation layers to evaluation mode
    with torch.no_grad():
        correct = 0
        for i, data in enumerate(test_loader):
            predicted_output = model1(data.x, data.edge_index, data.edge_attr, data.batch)
            target = data.y.unsqueeze(1).float()

            prediction = (m(predicted_output) >= 0.5).float()
            correct += int((prediction == target).sum())  # Check against ground-truth labels.

            batch_prob = m(predicted_output).tolist()
            batch_pred = prediction.tolist()
            batch_lab = data.y.tolist()

            lab.extend(batch_lab)
            pred.extend(batch_pred)
            prob.extend(batch_prob)

            test_loss = criterion(predicted_output, target)
            running_test_loss += test_loss.item()  # track the loss value

    test_accuracy = correct / len(test_loader.dataset)
    test_loss_value = running_test_loss / len(test_loader)
    pred = np.array(pred)
    lab = np.array(lab)
    prob = np.array(prob)

    # print('Classification')
    # print('Micro F1 score ', f1_score(y_true=lab, y_pred=pred, average='micro'))
    # print('Macro F1 score ', f1_score(y_true=lab, y_pred=pred, average='macro'))
    # print('Weighted F1 score ', f1_score(y_true=lab, y_pred=pred, average='weighted'))

    fpr, tpr, threshold = roc_curve(lab, prob)
    roc_auc = auc(fpr, tpr)

    cm = confusion_matrix(lab, pred)

    # print(cm)

    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]

    # calculate the accuracy
    accuracy = ((TP + TN) / float(TP + TN + FP + FN))

    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))

    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))

    # balanced accuracy
    bal_accuracy = (conf_specificity+conf_sensitivity) / 2

    # print("bal acc", bal_accuracy, "AUC", roc_auc, "sens", conf_sensitivity, "spc", conf_specificity)

    # # Visualize
    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')

    # plt.xlabel('Predictions', fontsize=18)
    # plt.ylabel('Actuals', fontsize=18)
    # plt.title('Confusion Matrix', fontsize=18)
    # plt.show()

    return  lab, roc_auc, tpr, prob, conf_sensitivity, conf_specificity, accuracy, bal_accuracy, test_loss_value, cm, prob, lab



def test_MIL(test_loader, model1, m, criterion):
    pred, lab, prob = [], [], []
    running_test_loss = 0.0
    model1.eval() #to set dropout and batch normalisation layers to evaluation mode
    with torch.no_grad():
        correct = 0
        for data, bagids, labels in test_loader:
            predicted_output =  model1((data, bagids))
            target = labels.unsqueeze(1).float()

            prediction = (m(predicted_output) >= 0.5).float()
            correct += int((prediction == target).sum())  # Check against ground-truth labels.

            batch_prob = m(predicted_output).tolist()
            batch_pred = prediction.tolist()
            batch_lab = labels.tolist()

            lab.extend(batch_lab)
            pred.extend(batch_pred)
            prob.extend(batch_prob)

            test_loss = criterion(predicted_output, target)
            running_test_loss += test_loss.item()  # track the loss value

    test_accuracy = correct / len(test_loader.dataset)
    test_loss_value = running_test_loss / len(test_loader)
    pred = np.array(pred)
    lab = np.array(lab)
    prob = np.array(prob)

    # print('Classification')
    # print('Micro F1 score ', f1_score(y_true=lab, y_pred=pred, average='micro'))
    # print('Macro F1 score ', f1_score(y_true=lab, y_pred=pred, average='macro'))
    # print('Weighted F1 score ', f1_score(y_true=lab, y_pred=pred, average='weighted'))

    fpr, tpr, threshold = roc_curve(lab, prob)
    roc_auc = auc(fpr, tpr)

    cm = confusion_matrix(lab, pred)

    # print(cm)

    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]

    # calculate the accuracy
    accuracy = ((TP + TN) / float(TP + TN + FP + FN))

    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))

    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))

    # balanced accuracy
    bal_accuracy = (conf_specificity+conf_sensitivity) / 2

    # print("bal acc", bal_accuracy, "AUC", roc_auc, "sens", conf_sensitivity, "spc", conf_specificity)

    # # Visualize
    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')

    # plt.xlabel('Predictions', fontsize=18)
    # plt.ylabel('Actuals', fontsize=18)
    # plt.title('Confusion Matrix', fontsize=18)
    # plt.show()

    return  lab, roc_auc, tpr, prob, conf_sensitivity, conf_specificity, accuracy, bal_accuracy, test_loss_value, cm, prob, lab

