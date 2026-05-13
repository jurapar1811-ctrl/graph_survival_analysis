import torch
import numpy as np
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
from scipy.optimize import curve_fit
from numpy import trapz



def kaplanmeyer_curves(y, pred, writer=None, test_s=-1, l=-1, n_gsearch=-1, test_or_all_pat='test'):
    y_formatted = np.array([(y[1][i],y[0][i]) for i in range(len(y[0]))], dtype=[('Event', '?'), ('Duration', '<f8')])#y[1],y[0] #np.array([(y[1][i],y[0][i]) for i in range(len(y))], dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    timepoints_pred_float = sorted(pred.columns.astype(float))

    estimated_event_time = np.array([timepoints_pred_float[i] for i in np.argmin(np.abs(0.5-pred.values), axis=1)]) 
    median_estimated_event_time = np.median(estimated_event_time)
  
    # Compute the area using the composite trapezoidal rule.
    area_under_surv_curve =  trapz(pred.values, x=timepoints_pred_float, axis=1) #AUSC
    median_area_under_surv_curve = np.median(area_under_surv_curve)

    group_indicator_event_time = np.where(estimated_event_time>=median_estimated_event_time, False, True) #False if event after median time, True otherwise
    group_indicator_AUSC =  np.where(area_under_surv_curve>=median_area_under_surv_curve, False, True) #False if AUSC over median time, True otherwise

    only_one_group_eventtime = True if(len(np.unique(group_indicator_event_time))==1) else False
    only_one_group_AUSC = True if(len(np.unique(group_indicator_AUSC))==1) else False 
    _, pvalue_event_time = compare_survival(y_formatted, group_indicator_event_time, return_stats=False) if(not only_one_group_eventtime) else (None, -1)
    _, pvalue_AUSC = compare_survival(y_formatted, group_indicator_AUSC, return_stats=False) if(not only_one_group_AUSC) else (None, -1)
# y (structured array, shape = (n_samples,)) – A structured array containing the binary event indicator (0=censoring) as first field, and time of event or time of censoring as second field.
# group_indicator (array-like, shape = (n_samples,)) – Group membership of each sample.

    if(writer!=None and n_gsearch==1):
        if(not only_one_group_eventtime):
            fig = plt.figure() 
            ax = fig.add_subplot(111)
            for group in ("event before median time", "event after median time"):             
                group_indicator = ~group_indicator_event_time if(group == 'event after median time') else group_indicator_event_time
                time_treatment, survival_prob_treatment, conf_int = kaplan_meier_estimator(event=y[1][group_indicator].astype(bool), time_exit=y[0][group_indicator], conf_type="log-log")
                ax.step(time_treatment, survival_prob_treatment, where="post", label=group)
                ax.fill_between(time_treatment, conf_int[0], conf_int[1], alpha=0.25, step="post")
            plt.text(2,0.03,f'p-value: {pvalue_event_time}')
            plt.ylim(0, 1)
            plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
            plt.xlabel("time $t$")
            plt.title("Kaplan-Meyer curves")
            plt.legend(loc="lower right")
            # plt.show()
            writer.add_figure(f"_KMcurve_eventtime_{test_or_all_pat}/test_loop_{test_s+1}_loop_{l+1}", fig, close=True)
        fig = plt.figure() 
        ax = fig.add_subplot(111)
        for group in ("AUSC under median", "AUSC over median"):             
            group_indicator = ~group_indicator_AUSC if(group == 'AUSC over median') else group_indicator_AUSC
            time_treatment, survival_prob_treatment, conf_int = kaplan_meier_estimator(event=y[1][group_indicator].astype(bool), time_exit=y[0][group_indicator], conf_type="log-log")
            ax.step(time_treatment, survival_prob_treatment, where="post", label=group)
            ax.fill_between(time_treatment, conf_int[0], conf_int[1], alpha=0.25, step="post")
        plt.text(2,0.03,f'p-value: {pvalue_AUSC}')
        plt.ylim(0, 1)
        plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
        plt.xlabel("time $t$")
        plt.title("Kaplan-Meyer curves")
        plt.legend(loc="lower right")
        # plt.show()
        writer.add_figure(f"_KMcurve_AUSC_{test_or_all_pat}/test_loop_{test_s+1}_loop_{l+1}", fig, close=True)

    return pvalue_event_time, pvalue_AUSC



def kaplanmeyer_curves_classif(y, binary_pred, writer=None, test_s=-1, l=-1, n_gsearch=-1, test_or_all_pat='test'):
    #Split patients depending on their predicted value (0 or 1)
    y_formatted = np.array([(y[1][i],y[0][i]) for i in range(len(y[0]))], dtype=[('Event', '?'), ('Duration', '<f8')])#y[1],y[0] #np.array([(y[1][i],y[0][i]) for i in range(len(y))], dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
  
    only_one_group = True if(len(np.unique(binary_pred))==1) else False
    _, pvalue = compare_survival(y_formatted, binary_pred, return_stats=False) if(not only_one_group) else (None, -1)
# y (structured array, shape = (n_samples,)) – A structured array containing the binary event indicator (0=censoring) as first field, and time of event or time of censoring as second field.
# group_indicator (array-like, shape = (n_samples,)) – Group membership of each sample.

    if(writer!=None and n_gsearch==1):
        if(not only_one_group):
            fig = plt.figure() 
            ax = fig.add_subplot(111)
            for group in ("low-risk", "high-risk"):             
                group_indicator = ~binary_pred if(group == 'low-risk') else binary_pred
                time_treatment, survival_prob_treatment, conf_int = kaplan_meier_estimator(event=y[1][group_indicator].astype(bool), time_exit=y[0][group_indicator], conf_type="log-log")
                ax.step(time_treatment, survival_prob_treatment, where="post", label=group)
                ax.fill_between(time_treatment, conf_int[0], conf_int[1], alpha=0.25, step="post")
            plt.text(2,0.03,f'p-value: {pvalue}')
            plt.ylim(0, 1)
            plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
            plt.xlabel("time $t$")
            plt.title("Kaplan-Meyer curves")
            plt.legend(loc="lower right")
            # plt.show()
            writer.add_figure(f"_KMcurve_classif_{test_or_all_pat}/test_loop_{test_s+1}_loop_{l+1}", fig, close=True)

    return pvalue



def get_hazard_from_survival_curve(pred: torch.tensor) -> torch.tensor:
    """
    Convert the survival curve predicted to hazard curve

    Args:
        timepoints_pred: (T) tensor with the time corresponding to each point of the survival curve predicted
        pred: (N,T) matrix with survival predictions

    Returns: tensor with hazard value
    """
    if(np.unique(pred.detach().numpy())[1]<1e-09): 
        print('on cape à trop haut pr le retour au risque instantané '+'*'*20)
        print(np.unique(pred.detach().numpy())[1])
        # raise ValueError('on cape à trop haut pr le retour au risque instantané')

    cum_hazard = -torch.log(torch.maximum(pred,torch.ones_like(pred)*1e-09)) #maximum to avoid infinite hazard values
    
    #As we have discret values of survival/cumulative hazard, the hazard rate is just the difference between adjacent elements of the cumulative hazard:
    return torch.diff(cum_hazard, prepend=torch.zeros((cum_hazard.shape[0],1)))



def visual_analysis_best_model_our_data(timepoints_pred_float, preds_before, preds_after, labels, writer=None):
    batch_size = 15

    #c-index
    fig, ax = plt.subplots(sharex=True,nrows=2,ncols=2, figsize=(20,20))    
    for i in range(batch_size):    
        ax[0,0].plot(timepoints_pred_float, preds_before[i], color='C'+str(i))
        ax[0,1].plot(timepoints_pred_float, preds_after.detach()[i], color='C'+str(i))
    ax[0,0].set_title("Initial prediction")
    ax[0,1].set_title("Corrected prediction")   
    for i in range(batch_size):    
        linestyles = '-' if (labels[i,1]==1.0) else '--' 
        # ax[1,0].axvline(x=[timepoints_pred_float[torch.where(torch.tensor(timepoints_pred_float) == round(labels[i,0].item(),4))[0]]], linestyle=linestyles, ymin=0, ymax=1, color='C'+str(i))
        # ax[1,1].axvline(x=[timepoints_pred_float[torch.where(torch.tensor(timepoints_pred_float) == round(labels[i,0].item(),4))[0]]], linestyle=linestyles, ymin=0, ymax=1, color='C'+str(i))
        ax[1,0].axvline(x=[labels[i,0]], linestyle=linestyles, ymin=0, ymax=1, color='C'+str(i))
        ax[1,1].axvline(x=[labels[i,0]], linestyle=linestyles, ymin=0, ymax=1, color='C'+str(i))

    ax[1,0].set_title("Event (full) / censoring (dotted) time")
    ax[1,0].set_xlabel("Time")
    ax[1,1].set_title("Event (full) / censoring (dotted) time")
    ax[1,1].set_xlabel("Time")
    plt.show()

    #bs  
    for i in range(batch_size):    
        fig, ax = plt.subplots(sharex=True,nrows=2,ncols=2, figsize=(20,20))  
        linestyles_1 = '-' if (labels[4*i,1]==1.0) else '--'
        linestyles_2 = '-' if (labels[4*i+1,1]==1.0) else '--'
        ax[0,0].plot(timepoints_pred_float, preds_before[4*i])
        ax[0,0].plot(timepoints_pred_float, preds_after.detach()[4*i])
        ax[0,1].plot(timepoints_pred_float, preds_before[4*i+1])
        ax[0,1].plot(timepoints_pred_float, preds_after.detach()[4*i+1]) 
        ax[0,0].axvline(x=[timepoints_pred_float[torch.where(torch.tensor(timepoints_pred_float) == round(labels[4*i,0].item(),4))[0]]], linestyle=linestyles_1, ymin=0, ymax=1)
        ax[0,1].axvline(x=[timepoints_pred_float[torch.where(torch.tensor(timepoints_pred_float) == round(labels[4*i+1,0].item(),4))[0]]], linestyle=linestyles_2, ymin=0, ymax=1)
        linestyles_1 = '-' if (labels[4*i+2,1]==1.0) else '--'
        linestyles_2 = '-' if (labels[4*i+3,1]==1.0) else '--'
        ax[1,0].plot(timepoints_pred_float, preds_before[4*i+2])
        ax[1,0].plot(timepoints_pred_float, preds_after.detach()[4*i+2])
        ax[1,1].plot(timepoints_pred_float, preds_before[4*i+3])
        ax[1,1].plot(timepoints_pred_float, preds_after.detach()[4*i+3]) 
        ax[1,0].axvline(x=[timepoints_pred_float[torch.where(torch.tensor(timepoints_pred_float) == round(labels[4*i+2,0].item(),4))[0]]], linestyle=linestyles_1, ymin=0, ymax=1)
        ax[1,1].axvline(x=[timepoints_pred_float[torch.where(torch.tensor(timepoints_pred_float) == round(labels[4*i+3,0].item(),4))[0]]], linestyle=linestyles_2, ymin=0, ymax=1)
        ax[0,0].set_title("Corrected prediction in orange")
        ax[0,1].set_title("Corrected prediction in orange") 
        ax[1,0].set_xlabel("Time")
        ax[1,1].set_xlabel("Time")  
        plt.show()

    # if writer!=None: writer.add_figure(f"analysis_cindex/test_loop_{i+1}_loop_{l+1}", fig)#, close=True)

    # if writer!=None: writer.add_figure(f"analysis_bs/test_loop_{i+1}_loop_{l+1}", fig)#, close=True)

    return 0


def visual_analysis_best_model(timepoints_pred_float, preds_before, preds_after, labels, writer=None):
    batch_size = 15

    #c-index
    fig, ax = plt.subplots(sharex=True,nrows=2,ncols=2, figsize=(20,20))    
    for i in range(batch_size):    
        ax[0,0].plot(timepoints_pred_float, preds_before[i], color='C'+str(i))
        ax[0,1].plot(timepoints_pred_float, preds_after.detach()[i], color='C'+str(i))
    ax[0,0].set_title("Initial prediction")
    ax[0,1].set_title("Corrected prediction")   
    for i in range(batch_size):    
        linestyles = '-' if (labels[i,1]==1.0) else '--' 
        ax[1,0].axvline(x=[timepoints_pred_float[torch.where(torch.tensor(timepoints_pred_float) == labels[i,0])[0]]], linestyle=linestyles, ymin=0, ymax=1, color='C'+str(i))
        ax[1,1].axvline(x=[timepoints_pred_float[torch.where(torch.tensor(timepoints_pred_float) == labels[i,0])[0]]], linestyle=linestyles, ymin=0, ymax=1, color='C'+str(i))

    ax[1,0].set_title("Event (full) / censoring (dotted) time")
    ax[1,0].set_xlabel("Time")
    ax[1,1].set_title("Event (full) / censoring (dotted) time")
    ax[1,1].set_xlabel("Time")
    plt.show()

    #bs  
    for i in range(batch_size):    
        fig, ax = plt.subplots(sharex=True,nrows=2,ncols=2, figsize=(20,20))  
        linestyles_1 = '-' if (labels[4*i,1]==1.0) else '--'
        linestyles_2 = '-' if (labels[4*i+1,1]==1.0) else '--'
        ax[0,0].plot(timepoints_pred_float, preds_before[4*i])
        ax[0,0].plot(timepoints_pred_float, preds_after.detach()[4*i])
        ax[0,1].plot(timepoints_pred_float, preds_before[4*i+1])
        ax[0,1].plot(timepoints_pred_float, preds_after.detach()[4*i+1]) 
        ax[0,0].axvline(x=[timepoints_pred_float[torch.where(torch.tensor(timepoints_pred_float) == labels[4*i,0])[0]]], linestyle=linestyles_1, ymin=0, ymax=1)
        ax[0,1].axvline(x=[timepoints_pred_float[torch.where(torch.tensor(timepoints_pred_float) == labels[4*i+1,0])[0]]], linestyle=linestyles_2, ymin=0, ymax=1)
        linestyles_1 = '-' if (labels[4*i+2,1]==1.0) else '--'
        linestyles_2 = '-' if (labels[4*i+3,1]==1.0) else '--'
        ax[1,0].plot(timepoints_pred_float, preds_before[4*i+2])
        ax[1,0].plot(timepoints_pred_float, preds_after.detach()[4*i+2])
        ax[1,1].plot(timepoints_pred_float, preds_before[4*i+3])
        ax[1,1].plot(timepoints_pred_float, preds_after.detach()[4*i+3]) 
        ax[1,0].axvline(x=[timepoints_pred_float[torch.where(torch.tensor(timepoints_pred_float) == labels[4*i+2,0])[0]]], linestyle=linestyles_1, ymin=0, ymax=1)
        ax[1,1].axvline(x=[timepoints_pred_float[torch.where(torch.tensor(timepoints_pred_float) == labels[4*i+3,0])[0]]], linestyle=linestyles_2, ymin=0, ymax=1)
        ax[0,0].set_title("Corrected prediction in orange")
        ax[0,1].set_title("Corrected prediction in orange") 
        ax[1,0].set_xlabel("Time")
        ax[1,1].set_xlabel("Time")  
        plt.show()

    # if writer!=None: writer.add_figure(f"analysis_cindex/test_loop_{i+1}_loop_{l+1}", fig)#, close=True)

    # if writer!=None: writer.add_figure(f"analysis_bs/test_loop_{i+1}_loop_{l+1}", fig)#, close=True)

    return 0




def analysis_bs(surv_test_pat, label_test_pat, timepts_pred):
    # print(type(surv_test_pat)) #df
    # print(surv_test_pat.shape) #MLP: 1631,887
    label_test_pat_clean = torch.tensor([label_test_pat[0],label_test_pat[1]])
    # print(type(label_test_pat_clean)) #tensor
    # print(label_test_pat_clean.shape) #MLP: 2,887

    print(surv_test_pat.head())

    for t in timepts_pred:
        if(t>1000):
            print(t)
            sum_survival_at_risk_pat = np.sum(surv_test_pat.loc[t, np.where(round(label_test_pat_clean[0,:],4)>t)].values)
            sum_survival_event_passed_pat = np.sum(surv_test_pat.loc[t, np.where([t>=round(label_test_pat_clean[0,:],4) and label_test_pat_clean[1,i]==1 for i in range(len(label_test_pat_clean[0]))])].values)
            print(sum_survival_at_risk_pat, sum_survival_event_passed_pat)
    print(a)

    return 0



#fitting curve to survival curve obtained
def sigmoid(x, L ,x0, k, b):
    """
    Inputs:
        x is the data to fit
        L is responsible for scaling the output range from [0,1] to [0,L]
        b adds bias to the output and changes its range from [0,L] to [b,L+b]
        k is responsible for scaling the input, which remains in (-inf,inf)
        x0 is the point in the middle of the Sigmoid, i.e. the point where Sigmoid should originally output the value 1/2 [since if x=x0, we get 1/(1+exp(0)) = 1/2].
    """
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

def exponential(x, L ,x0, k, b):
    y = L*np.exp(k*(x-x0)) + b
    return(y)

def fitting_curve(xdata, ydata, curve='sigmoid', min_vals=0, max_vals=1):

    p0 = [max(ydata), np.median(xdata),1,min(ydata)] # this is an mandatory initial guess

    if(curve=='sigmoid'):
        curve_to_fit = sigmoid
    elif(curve=='exp'):
        curve_to_fit = exponential
    else:
        raise ValueError(f'"curve": {curve} in fitting_curve() in survival.py not implemented')
    popt, pcov = curve_fit(curve_to_fit, xdata, ydata, p0, bounds=(min_vals,max_vals), method='dogbox')
    return sigmoid(xdata, *popt)