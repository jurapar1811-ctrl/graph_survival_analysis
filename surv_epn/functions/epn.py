"""
Description: This file is used to define the Error Passing Network model.
"""
from typing import List, Optional, Tuple

import sys
import math
import torch
import pycox
import datetime
import numpy as np
import pandas as pd
import torchtuples as tt
import matplotlib.pyplot as plt
from torch import cat, eye, matmul, no_grad, sqrt, Tensor, unique
from torch.nn import Linear, MSELoss
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from main.parameters_EPN import parameters
###
sys.path.insert(1, '../') 
sys.path.insert(1, './') 
from functions.survival import get_hazard_from_survival_curve


# from src.data.processing.datasets import MaskType, PetaleDataset
# from src.evaluation.early_stopping import EarlyStopper
# from src.models.abstract_models.custom_torch_base import TorchCustomModel
# from src.models.wrappers.torch_wrappers import TorchRegressorWrapper
# from src.utils.hyperparameters import HP, NumericalContinuousHP, NumericalIntHP
# from src.utils.metrics import RootMeanSquaredError


# class SimilarityKernel:
#     """
#     Stores the constant related to mask types
#     """
#     ATTENTION: str = "attention"
#     DOT: str = "dot_product"
#     COSINE: str = "cosine"

#     def __iter__(self):
#         return iter([self.ATTENTION, self.DOT, self.COSINE])

#np.set_printoptions(formatter={'float':           "{0:0.9f}".format})        "{0:0.9f}".format})
#torch.set_printoptions(precision=10)


#Put outside otherwise we can't save our model with torch ("AttributeError: Can't pickle local object 'EPN._define_similarity_kernel.<locals>.compute_similarity'")
def compute_similarity(similarity_kernel, _key_projection, _query_projection, _dk, x1: Tensor, x2: Tensor, _print=-1) -> Tensor:
    if similarity_kernel == 'att':
        return matmul(_key_projection(x1), _query_projection(x2).t()) / _dk
    elif similarity_kernel == 'dot':
        return matmul(x1, x2.t())
    elif similarity_kernel == 'cosine':
        # if(_print[0]!=-1):print(matmul((x1/torch.norm(x1, dim=1, keepdim=True))[_print[0]],((x2/torch.norm(x2, dim=1, keepdim=True))[1]).t()))
        # if(_print[0]!=-1):print(matmul(x1 / torch.norm(x1, dim=1, keepdim=True),(x2 / torch.norm(x2, dim=1, keepdim=True)).t())[_print[0]][1])
        return matmul(x1 / torch.norm(x1, dim=1, keepdim=True),
                        (x2 / torch.norm(x2, dim=1, keepdim=True)).t())
    elif similarity_kernel == 'att_neg': #Here, key_projection is the matrix A and query_projection the matrix V
        raise ValueError('not implemented  yet')
        print(x1.shape)
        print(_query_projection(x1).shape)
        return torch.tanh(_key_projection(torch.concat(_query_projection(x1), _query_projection(x2))))
    elif similarity_kernel == 'diff': #Here, key_projection is the matrix A and query_projection the matrix 
        return 1/(torch.abs(x1.repeat(1,len(x2))-x2.t().repeat(len(x1),1))+1e-5) #1e-1 to avoid dividing by 0: should be adapted to the differences we have
    else:
        raise ValueError(f'Similarity kernel: {similarity_kernel} not implemented')
    


class EPN(torch.nn.Module): 
    """
    Error Passing Network model
    """

    def __init__(self,
                #  previous_pred_idx: int,
                #  pred_mu: float,
                #  pred_std: float,
                #  alpha: float = 0,
                #  beta: float = 0,
                #  num_cont_col: Optional[int] = None,
                #  cat_idx: Optional[List[int]] = None,
                #  cat_sizes: Optional[List[int]] = None,
                #  cat_emb_sizes: Optional[List[int]] = None,
                #  verbose: bool = False,
                 _parameters: parameters,
                 class_weight: float = 0.5,
                 criterion = torch.nn.BCELoss(),
                 alpha: float = 0,
                 beta: float = 0,
                 input_size: int = 0,
                 similarity_kernel: str = 'att',
                 prop_neighbors: int = None):
        
        super(EPN, self).__init__()
        
        # Type of features to use
        self._feats_for_popg = _parameters.feats_for_popg
        self._feats_usable = _parameters.feats_usable
        self._feats_usable_dim = _parameters.feats_usable_dim

        # Model input dimension
        self._input_size = input_size

        # Number of neighbors and similarity metric
        self._prop_neighbors = prop_neighbors
        self._similarity_kernel = similarity_kernel

        # Scaling factor
        self._dk = sqrt(Tensor([self._input_size]))#*100

        # Attention map cache
        self._ind_cache = None
        self._attn_cache = None

        # Key and Query projection layers (if needed for the attention process)
        # We decrease the input size by one because one column contains predicted targets and will be removed ####################### REMOVED
        self._key_projection = Linear(self._input_size, self._input_size)
        self._query_projection = Linear(self._input_size, self._input_size)

        # Projection layers for attention computation as in Zhao, W., Nakashima, Y., Chen, H., Babaguchi, N., 2024. 
        # Enhancing Fake News Detection in Social Media via Label Propagation on Cross-modal Tweet Graph.
        self._A_projection = Linear(2*self._input_size, 2)
        self._V_projection = Linear(self._input_size, self._input_size)
        # self._V2_projection = Linear(self._input_size, self._input_size)
        self._temperature_att = 50 #No implemented there

        # Loss function
        self._criterion = criterion #pos_weight=class_weight) #BCEWithLogitsLoss(pos_weight=class_weight)
        self._alpha = alpha
        self._beta = beta


    @property
    def attn_cache(self) -> Tensor:
        return self._attn_cache




    def forward(self,
                input_db: Tensor,
                training: bool = False,
                test_idx: Optional[List[int]] = None,
                full_test_idx: Optional[List[int]] = None,
                id_pat_pb: int=-1) -> Tensor:
        """
        Executes a forward pass.

        Args:
            input_db: database including the patients' ids, their feature vector for the graph construction, and their label and prediction at least
            training: boolean indicating whether we're training or testing our model
            test_idx: List of idx associated to test data points for which we want to calculate smooth targets.
                      If None, values are returned for all idx.
            full_test_idx: List of idx associated to all patients in the test set even if they are not ni the current balanced subset tested on.
                           Sould be None for training. Not taken into account if test_idx=None.         
            id_pat_pb: only used for debug 

        Returns: (N, 1) tensor with smoothed targets
        """

        # We extract previous prediction made by another model   #-> we have them directly as inputs now
        preds = torch.from_numpy(input_db['pred'].values).float()
        # preds = [input_db[input_db['pat_id']==pat] for pat in ]
        #y_hat = (features[:, self._prediction_idx] * self._pred_std) + self._pred_mu   #Our predictions haven't been standardized

        # We calculate the errors made on these predictions
        labels = torch.from_numpy(input_db['label'].values).float()
        errors = labels - preds


        # #############################################################
        # We retrieve the features used to compute the attention between patients
        input_db_dropped = input_db.drop(['index','pat_id','split','pred','label'], axis=1)
        j=0
        for feat in self._feats_usable:
            if(feat not in self._feats_for_popg):
                input_db_dropped = input_db_dropped.drop([f'{feat}_{i+1}' for i in range(self._feats_usable_dim[j])], axis=1)
            j+=1

        features = torch.from_numpy(input_db_dropped.values).float() #[patient][feat]

        # # We initialize a list of tensors to concatenate
        # new_features = []

        # # We extract continuous data
        # if len(self._cont_idx) != 0:
        #     new_features.append(features[:, [i for i in self._cont_idx if i != self._prediction_idx]])

        # # We perform entity embeddings on categorical features
        # if len(self._cat_idx) != 0:
        #     new_features.append(self._embedding_block(x))

        # # We concatenate all inputs
        # x = cat(new_features, 1)
        # #############################################################   We have all that at once as input

        if(self._similarity_kernel == 'att_neg'):

            # We compute the attention, positive or negative, as in Zhao, W., Nakashima, Y., Chen, H., Babaguchi, N., 2024. 
            # Enhancing Fake News Detection in Social Media via Label Propagation on Cross-modal Tweet Graph.
            att = compute_similarity(self._similarity_kernel, self._A_projection, self._V_projection, self._dk, features, features)
            

        if test_idx is None:

            # We compute the scaled-dot product
            att = compute_similarity(self._similarity_kernel, self._key_projection, self._query_projection, self._dk, features, features)

            # We set the diagonal to zero
            # att = att * (1 - eye(att.shape[0], att.shape[1]))
            att = att - 2*eye(att.shape[0], att.shape[1])*torch.finfo(torch.float32).max # to the minimal value to avoid getting an attention after the softmax  #MODIF
                #No test idx: we do not take into account full_test_idx 

            # Get the k neighbors
            if self._prop_neighbors != None:
                att, indices = torch.topk(att, k=max(round(self._prop_neighbors*att.shape[1]), 1), dim=-1)
                errors = errors[indices]
                self._ind_cache = indices
                # print(errors)

            else:
                # Reshape the dataset to len(x) * number of total nodes
                errors = errors.unsqueeze(1).repeat(1, len(features)).permute(1, 0)

            # We apply the softmax
            self._attn_cache = att = softmax(att, dim=-1)

        else:
            id_id_pat_pb = np.where(test_idx==id_pat_pb)[0] #just for debug (np would be best?)

            # We compute the scaled-dot product
            att = compute_similarity(self._similarity_kernel, self._key_projection, self._query_projection, self._dk, features[test_idx, :], features) if(id_pat_pb!=-1) else compute_similarity(self._similarity_kernel, self._key_projection, self._query_projection, self._dk, features[test_idx, :], features, [id_id_pat_pb,id_pat_pb])            
            

            if not training:

                # We set the attention given to all test elements to zero
                # Therefore test elements cannot attend to errors of other test elements (including their own)
                att[:, full_test_idx] = torch.finfo(torch.float32).min  # rather than 0, put to the minimal value to avoid getting an attention after the softmax #MODIF
                    #We cannot pay attention to ANY (full_test_idx instead of test_idx) test element, even to those not in the current balanced set

            else:

                # We only make sure that self attention value of test elements (so all elements as we are training) are zeroedzeroed
                att[range(len(test_idx)), test_idx] = torch.finfo(torch.float32).min  # rather than 0, put to the minimal value to avoid getting an attention after the softmax  #MODIF
                    #Training: thus, no need for full_test_idx

            # Get the k neighbors
            if self._prop_neighbors != None:
                att, indices = torch.topk(att, k=max(round(self._prop_neighbors*att.shape[1]), 1), dim=1) #[batch size][n_neighbors]
                errors = errors[indices] #The error of the k neighbors to which we pay the most important attention
                self._ind_cache = indices
                # print(errors)

                #only to study the behavior of the system:
                mat_right_neighbor = [[0.5+2*(labels[test_idx[i]].item()-0.5)*(labels[indices[i][j]].item()-0.5) for i in range(len(test_idx))] for j in range(len(indices[0]))]
                # if not training: 
                # if not training: print(np.transpose(mat_right_neighbor))

            else:
                # Reshape the dataset to test_idx * number of total nodes
                errors = errors.unsqueeze(1).repeat(1, len(test_idx)).permute(1, 0)

            pat_for_test = 0
            # if not training: print(self._key_projection(features[pat_for_test,:]))
            # if not training: print(self._query_projection(features[indices[pat_for_test,0],:]))
            # if not training: print(self._query_projection(features[indices[pat_for_test,1],:]))
            # if not training: print(self._query_projection(features[indices[pat_for_test,-1],:]))

            # if not training:
            #     print('att')
            #     print(att)
            # if not training: print(input_db['pat_id'].values[indices[:,0]])
            # if not training: print(input_db['label'].values[test_idx])

            # We apply the softmax
            self._attn_cache = att = softmax(att, dim=-1)

            # We only keep predictions previously made for test idx
            preds = preds[test_idx]

        return torch.diagonal(matmul(att, errors.t()), 0)+ preds  #(matmul(att, errors.t())[:, 0] + preds).squeeze(dim=-1)
                ###################### /!\ (otherwise pb for K-NN option)
    

    def loss(self,
             pred: torch.tensor,
             y: torch.tensor) -> torch.tensor:
        """
        Calls the criterion and add the elastic penalty (for binary classification)

        Args:
            pred: (N,) tensor with predictions
            y: (N,) tensor with targets

        Returns: tensor with loss value
        """
        # Computations of penalties
        l1_penalty, l2_penalty = torch.tensor(0.), torch.tensor(0.)
        for _, w in self.named_parameters():
            l1_penalty = l1_penalty + w.abs().sum()
            l2_penalty = l2_penalty + w.pow(2).sum()
        
        # Class weight adaptation
        #ratio_classes = torch.count_nonzero(torch.where(y==0, 1, 0))/torch.count_nonzero(torch.where(y==1, 1, 0))
        #print(ratio_classes)
        ratio_classes = 5 #Global ratio of negative to positive patients around 5
        preds_class0 = pred[torch.where(y==0, 1, 0).nonzero().squeeze()]
        preds_class1 = pred[torch.where(y==1, 1, 0).nonzero().squeeze()]
        pred = torch.cat((preds_class0, torch.tile(preds_class1, (math.floor(ratio_classes),))))
        try:
            y = torch.cat((torch.zeros(len(preds_class0)), torch.tile(torch.ones(len(preds_class1)), (math.floor(ratio_classes),))))
        except TypeError:
            y = torch.cat((torch.zeros(len(preds_class0)), torch.tile(torch.ones(1), (math.floor(ratio_classes),))))

        # Computation of loss reduction + elastic penalty
        return self._criterion(pred, y) + self._alpha * l1_penalty + self._beta * l2_penalty




###################################################################################################################################################################################
class EPN_surv(torch.nn.Module): 
    """
    Error Passing Network model

    n_control: number of control samples to use to compute the loss with case-control strategy
    """

    def __init__(self,
                #  previous_pred_idx: int,
                #  pred_mu: float,
                #  pred_std: float,
                #  alpha: float = 0,
                #  beta: float = 0,
                #  num_cont_col: Optional[int] = None,
                #  cat_idx: Optional[List[int]] = None,
                #  cat_sizes: Optional[List[int]] = None,
                #  cat_emb_sizes: Optional[List[int]] = None,
                #  verbose: bool = False,
                 _parameters: parameters,
                 class_weight: float = 0.5,
                 criterion = pycox.models.loss.CoxCCLoss(shrink=0.),
                 temp_att:float = 1,
                 alpha: float = 0,
                 beta: float = 0,
                 batch_size: int = 256,
                 n_control: int = 1,
                 input_size: int = 0,
                 similarity_kernel: str = 'att',
                 prop_neighbors: int = None):
        
        super(EPN_surv, self).__init__()
        
        # Type of features to use
        self._feats_for_popg = _parameters.feats_for_popg
        self._feats_usable = _parameters.feats_usable_surv
        self._feats_usable_dim = _parameters.feats_usable_surv_dim

        # Model input dimension
        self._input_size = input_size  #TEST FOR LABELS AS FEATURES: 1

        # Number of neighbors and similarity metric
        self._prop_neighbors = prop_neighbors
        self._similarity_kernel = similarity_kernel

        # Scaling factors
        self._dk = sqrt(Tensor([self._input_size]))#*100
        self._temperature_att = temp_att

        # Attention map cache
        self._ind_cache = None
        self._attn_cache = None

        # Key and Query projection layers (if needed for the attention process)
        # We decrease the input size by one because one column contains predicted targets and will be removed ####################### REMOVED
        self._key_projection = Linear(self._input_size, self._input_size)
        self._query_projection = Linear(self._input_size, self._input_size)

        # # Projection layers for attention computation as in Zhao, W., Nakashima, Y., Chen, H., Babaguchi, N., 2024. 
        # # Enhancing Fake News Detection in Social Media via Label Propagation on Cross-modal Tweet Graph.
        # self._A_projection = Linear(2*self._input_size, 2)
        # self._V_projection = Linear(self._input_size, self._input_size)
        # # self._V2_projection = Linear(self._input_size, self._input_size)

        # Loss function
        self._criterion = criterion #pos_weight=class_weight) #BCEWithLogitsLoss(pos_weight=class_weight)
        self._alpha = alpha
        self._beta = beta
        self._batch_size = batch_size
        self._n_control = n_control
        



    @property
    def attn_cache(self) -> Tensor:
        return self._attn_cache




    def forward(self,
                input_db: Tensor,
                # training: bool = False,
                # test_idx: Optional[List[int]] = None,
                # full_test_idx: Optional[List[int]] = None,
                #id_pat_pb: int=-1
                ) -> Tensor:
        """
        Executes a forward pass.

        Args:
            input_db: database including the patients' ids, their feature vector for the graph construction, and their label and prediction at least 

        Returns: (N, 1) tensor with smoothed targets
        """
            # training: boolean indicating whether we're training or testing our model
            # test_idx: List of idx associated to test data points for which we want to calculate smooth targets.
            #           If None, values are returned for all idx.
            # full_test_idx: List of idx associated to all patients in the test set even if they are not ni the current balanced subset tested on.
            #                Sould be None for training. Not taken into account if test_idx=None.         
            # id_pat_pb: only used for debug


        # We extract previous prediction made by another model   #-> we have them directly as inputs now
        # and we calculate the errors made on these predictions
        # if(self._feats_for_popg == 'METABRIC' or self._feats_for_popg == 'SUPPORT'):
        if(self._feats_for_popg == 'METABRIC' or self._feats_for_popg == 'SUPPORT'):
            timepoints_pred_float = sorted(input_db.columns.difference([input_db.columns[i] for i in range(len(input_db.columns)) if 'feat_' in input_db.columns[i]]\
                +['label_duration','label_event','split','pat_to_compute','batch','pat_id','index']).values.astype(float)) #'index' comes from the reset_index() operation
                #important to sort, because otherwise it returns '10', '100', '1000', '20', ... (which is absolutely not the order we expect) (example from SUPPORT)
        else:
            cols_all_feat = [f'{self._feats_usable[feat_nb]}_{i+1}' for feat_nb in range(len(self._feats_usable)) for i in range(self._feats_usable_dim[feat_nb])]
            cols_feat = [f'{self._feats_usable[feat_nb]}_{i+1}' for feat_nb in range(len(self._feats_usable)) for i in range(self._feats_usable_dim[feat_nb]) if(self._feats_usable[feat_nb] in self._feats_for_popg)]
            timepoints_pred_float = sorted(input_db.columns.difference(cols_all_feat+['label_duration','label_event','split','pat_to_compute','batch','pat_id','index']).values.astype(float)) #'index' comes from the reset_index() operation
                #important to sort, because otherwise it returns '10', '100', '1000', '20', ... (which is absolutely not the order we expect) (example from SUPPORT)
        timepoints_pred_str = [str(timepoints_pred_float[i]) for i in range(len(timepoints_pred_float))]
        preds = torch.from_numpy(input_db[timepoints_pred_str].values).float()    #[7098, 1631] #[448, 375] #[nb of patients in dataset_to_send, number of survival points]        
        labels = torch.from_numpy(input_db[['label_duration','label_event']].values).to(torch.float)  #[7098,2] #[number of patients in dataset_to_send, (duration, event)]
        errors = self.compute_error_survival(torch.tensor(timepoints_pred_float), preds, labels) #[nb of patients in dataset_to_send, number of survival points]     



        # #############################################################
        # We retrieve the features used to compute the attention between patients
        if(self._feats_for_popg == 'METABRIC' or self._feats_for_popg == 'SUPPORT'):
            features = torch.from_numpy(input_db[[input_db.columns[i] for i in range(len(input_db.columns)) if 'feat' in input_db.columns[i]]].values).float() #[patient][feat]
            # ####### TEST: USE LABELS AS FEATURES (METABRIC/SUPPORT)
            # features = torch.from_numpy(input_db['label_duration'].values).unsqueeze(1).float()
        else: 
            features = torch.from_numpy(input_db[cols_feat].values).float() #[patient][feat]

        test_idx = np.where(input_db.eval('pat_to_compute and batch'))[0] 
        # print('test_idx (and input_db): check good values (and always len()=batch size)')
        # print(test_idx)
        # print(len(test_idx))

        training = (len(np.where(input_db.eval('pat_to_compute and (split=="val" or split=="test")'))[0])==0)
        # print('training', training)
            

        # We compute the scaled-dot product
        att = compute_similarity(self._similarity_kernel, self._key_projection, self._query_projection, self._dk, features[test_idx, :], features) 


        if not training:

            full_test_idx = np.where(input_db['split'].values=='val')[0] if(len(np.where(input_db.eval('pat_to_compute and split=="test"')))==0) else np.where(input_db['split'].values=='test')[0]

            # We set the attention given to all test elements to zero
            # Therefore test elements cannot attend to errors of other test elements (including their own)
            att[:, full_test_idx] = torch.finfo(torch.float32).min  # rather than 0, put to the minimal value to avoid getting an attention after the softmax #MODIF
                #We cannot pay attention to ANY (full_test_idx instead of test_idx) test element, even to those not in the current balanced set

        else:

            # We only make sure that self attention value of test elements (so all elements as we are training) are zeroed
            att[range(len(test_idx)), test_idx] = torch.finfo(torch.float32).min  # rather than 0, put to the minimal value to avoid getting an attention after the softmax  #MODIF
                #Training: thus, no need for full_test_idx



        # #do not pay attention to censored patients:
        # censored_mask = torch.tensor(pd.eval('input_db.label_event == 0'))
        # att[:, censored_mask] = torch.finfo(torch.float32).min 
        




        # Get the k neighbors
        if self._prop_neighbors != -1:
            att, indices = torch.topk(att, k=max(round(self._prop_neighbors*att.shape[1]), 1), dim=1) #[batch size][n_neighbors]
            errors = errors[indices] #The error of the k neighbors to which we pay the most important attention #[batch size][n_neighbors_KNN][n_timepoints]
            self._ind_cache = indices

        else:
            # Reshape the errors to test_idx * number of total nodes
            errors = errors.unsqueeze(1).repeat(1, len(test_idx), 1).permute(1, 0, 2) #[batch size][n_neighbors][n_timepoints]

        # We apply the softmax
        self._attn_cache = att = softmax(att*self._temperature_att, dim=-1)#*4 #, dim=-1) #*self._temperature_att, dim=-1)




        # #do not correct censored patients:
        # db_test = input_db.iloc[test_idx]
        # censored_mask = torch.tensor(pd.eval('db_test.label_event == 0').values)
        # att[censored_mask, :] = 0




        # We only keep predictions previously made for test idx
        preds = preds[test_idx] #[batch size][number of survival timepoints]
        # print('t7')
        # print(datetime.datetime.today())
    
        # print('computation of the updated predictions: check that the operation makes sense (and test precisely on one patient)')
        # # print(att.shape)
        # print(preds)
        # print(errors.shape)
        # # print(att.unsqueeze(2).repeat(1,1,errors.shape[-1]).shape)
        # print(att)
        # print(errors[0,:5])
        # print(errors)
        # print(matmul(att, errors).shape)#.t()))
        # print(matmul(att, errors))#.t()))
        # # print(matmul(att.unsqueeze(2).repeat(1,1,errors.shape[-1]), errors).shape)#.t()))
        # # print(torch.diagonal(matmul(att, errors),0).shape)#.t()), 0))
        # print(torch.diagonal(matmul(att, errors),0).t()+ preds)
        # print((torch.diagonal(matmul(att, errors),0).t()+ preds)[0])
        # fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(12,5))        
        # ax[0].plot(timepoints_pred_float, preds[0])
        # ax[0].set_title("Initial prediction")
        # ax[0].set_xlabel("Time")
        # ax[1].plot(timepoints_pred_float, (torch.diagonal(matmul(att, errors),0).t()+ preds)[0])
        # ax[1].set_title("Corrected prediction")
        # ax[1].set_xlabel("Time")
        # plt.show()

        # fig, ax = plt.subplots()  
        # ax.plot(timepoints_pred_float, preds[0], label='Initial prediction')
        # ax.set_xlabel("Time")
        # ax.plot(timepoints_pred_float, (torch.diagonal(matmul(att, errors),0).t()+ preds)[0], label='Corrected prediction')
        # ax.legend()
        # plt.show()

        # print(errors.a)
        
        return torch.diagonal(matmul(att, errors), 0).t()+ preds  #(matmul(att, errors.t())[:, 0] + preds).squeeze(dim=-1) #preds - 0.1
                ###################### /!\ (otherwise pb for K-NN option)
    #Should return as many survival curves as there are 'pat_to_compute'+'batch' patients
    

    def compute_error_survival(self,
                               timepoints_pred: torch.tensor,
                               pred: torch.tensor,
                               y: torch.tensor) -> torch.tensor:
        """
        Compute the error between a predicted survival curve and the corresponding patient's label (with or without censoring)

        Args:
            timepoints_pred: (T) tensor with the time corresponding to each point of the survival curve predicted
            pred: (N,T) matrix with survival predictions
            y: (N,2) matrix with targets (duration and event)

        Returns: tensor with error value
        """
        # Create a continuous label to compare it with the continuous survival curve predicted
        continuous_label = torch.where(timepoints_pred.repeat(y.shape[0],1) > y[:,0].repeat(timepoints_pred.shape[0],1).transpose(0,1), 0, 1) #=0 when the event happened and 1 beforehand
            #[number of patients, number of timepoints]

        # Take into account the censoring
        continuous_label = torch.where((y[:,1]==1.0).repeat(timepoints_pred.shape[0],1).transpose(0,1), continuous_label, torch.where(continuous_label.to(bool), 1, torch.nan)) #=nan when the censoring (and not an event) happened (=> event=0)
            #[number of patients, number of timepoints] (1 when no event/censoring happened; 0 when event happened; nan when censoring happened)

        return torch.where(continuous_label.isnan(), 0, continuous_label-pred) #We enforce the error to be 0 (no correction) for patients when they are censored





    


    

    def loss(self,
             timepoints_pred: torch.tensor,
             pred: torch.tensor, #should be only predictions for patients we want to compute the loss for (or to use as control if they are censored)
                                 #we do not consider either as case or control patients that are not in the current batch and to be computed (i.e. in the right split)
             y: torch.tensor, #should be labels of the same patients as pred
             GAINED_db:bool=False) -> torch.tensor:
        """
        Calls the criterion and add the elastic penalty (for survival analysis)

        Args:
            timepoints_pred: (T) tensor with the time corresponding to each point of the survival curve predicted
            pred: (N,T) matrix with survival predictions
            y: (N,2) matrix with targets (duration and event)
            GAINED_db: boolean indicating whether we're using the GAINED database (True) or not (False) (default: False)

        Returns: tensor with loss value
        """
            # pred: (N,T) matrix with survival predictions
            # y: (N,2) matrix with targets (duration and event)
            # dataset: dataframe with input features and labels for all patients

        # Computations of penalties
        l1_penalty, l2_penalty = torch.tensor(0.), torch.tensor(0.)
        for _, w in self.named_parameters():
            l1_penalty = l1_penalty + w.abs().sum()
            l2_penalty = l2_penalty + w.pow(2).sum()        

        # Computation of the main survival loss 
            #(from compute_metric (cox_cc.py) and CoxCCLoss.forward (loss.py) (+make_dataloader_predict (cox_time.py)))  
        pred_hazard = get_hazard_from_survival_curve(pred)

        # # To observe the different curves if needed:
        # for i in range(256):
        #     pat_to_look_at= i
        #     fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(12,5))        
        #     ax[0].plot(timepoints_pred, pred[pat_to_look_at].detach().numpy())
        #     ax[0].set_title("Survival prediction")
        #     ax[0].set_xlabel("Time")
        #     ax[1].plot(timepoints_pred, pred_hazard[pat_to_look_at].detach().numpy())
        #     ax[1].set_title("Hazard prediction")
        #     ax[1].set_xlabel("Time")
        #     plt.show()
        
        g_case, g_control = self.make_case_control(timepoints_pred, pred_hazard, y, n_control=self._n_control, GAINED_db=GAINED_db) #in all of them, only pred has a grad: ok
            #self.make_dataloader(dataset, batch_size=self._batch_size, shuffle=True, num_workers=0, n_control=self._n_control)
        # main_loss=0
        # print('len(dataloader): should be 1 in theory: to validate')
        # print(len(dataloader))
        # for data in dataloader: #Should be useless (only =one) because we get one batch each time this function is run
        #     data = self._to_device(data)
        #     batch_size = data.lens().flatten().get_if_all_equal()
        #     if batch_size is None:
        #         raise RuntimeError("All elements in input does not have the same length.")
        #     case, control = data # both are dataframes with the right 'batch's set   #TupleTree
        #     # input_all = tt.TupleTree((case,) + control).cat() #undoable because of our dataframe format?
        #     # g_all = self.net(*input_all)

        # #Set the network output to the right format for the loss computation (g_case=tensor and g_control=tupletree of tensors)
        #     g_case = self.forward(case) 
        #     g_control = self.forward(control)
            
        #     g_all = tt.tuplefy(g_all).split(batch_size).flatten()
        #     g_case = g_all[0]
        #     g_control = g_all[1:]
        main_loss = self._criterion(g_case, g_control) #case and ctrl do not have grad

        # Computation of loss reduction + elastic penalty
        return main_loss + self._alpha * l1_penalty + self._beta * l2_penalty
    

    @staticmethod
    def _sorted_input_target(input, target):
        durations, _ = target.t()
        idx_sort = np.argsort(durations)
        if (idx_sort == np.arange(0, len(idx_sort))):#.all():
            return input, target
        input = tt.tuplefy(input).iloc[idx_sort] #tuple with a tensor of shape [batch size][nb timepoints survival] #print(input[0].shape)
        target = tt.tuplefy(target).iloc[idx_sort] #tuple with a tensor of shape [batch size][label size (=2)] #print(target[0].shape)        
        return input, target
    


    @staticmethod
    def make_at_risk_dict(durations):
        """Create dict(duration: indices) from sorted df.
        A dict mapping durations to indices.
        For each time => index of all individual alive.
        
        Arguments:
            durations {np.array} -- durations.
        """
        assert type(durations) is np.ndarray, 'Need durations to be a numpy array'
        durations = pd.Series(durations)
        assert durations.is_monotonic_increasing, 'Requires durations to be monotonic'
        allidx = durations.index.values
        keys = durations.drop_duplicates(keep='first')
        at_risk_dict = dict()
        for ix, t in keys.items(): #iteritems(): #CHANGED ORIANE
            at_risk_dict[t] = allidx[ix:]
        return at_risk_dict


    @staticmethod
    def sample_alive_from_dates(dates, at_risk_dict, n_control=1):
        '''Sample index from living at time given in dates.
        dates: np.array of times (or pd.Series).
        at_risk_dict: dict with at_risk_dict[time] = <array with index of alive in X matrix>.
        n_control: number of samples.
        '''
        lengths = np.array([at_risk_dict[x].shape[0] for x in dates])  # Can be moved outside
        idx = (np.random.uniform(size=(n_control, dates.size)) * lengths).astype('int')
        samp = np.empty((dates.size, n_control), dtype=int)#float)#int) #CHANGED ORIANE #CHANGED BACK
        # samp.fill(np.nan)
        # samp = samp.astype(int) #ADDED ORIANE

        for it, time in enumerate(dates):
            samp[it, :] = at_risk_dict[time][idx[:, it]]
        # print(np.count_nonzero(np.isnan(samp)))
        return samp
    

    def make_case_control(self,
             timepoints_pred: torch.tensor,
             pred_hazard: torch.tensor, #should be only predictions for patients we want to compute the loss for (or to use as control if they are censored)
                                 #we do not consider either as case or control patients that are not in the current batch and to be computed (i.e. in the right split)
             y: torch.tensor, #should be labels of the same patients as pred
             n_control:int=1,
             GAINED_db:bool=False) -> torch.tensor: 
        """
        Calls the criterion and add the elastic penalty (for survival analysis)

        Args:
            timepoints_pred: (T) tensor with the time corresponding to each point of the hazard curve predicted
            pred_hazard: (N,T) matrix with hazard curve predictions
            y: (N,2) matrix with targets (duration and event)
            n_control: number of control samples to consider for one case elemnt in the case-control strategy (default: 1)
            GAINED_db: boolean indicating whether we're using the GAINED database (True) or not (False) (default:False)

        Returns: tensor with loss value
        """
        # print('test of loss function')
        # print(timepoints_pred)
        # print(y)
        # print(np.unique(y[:,1], return_counts=True))

        #sort patients by event time
        sorted_pred, sorted_y = self._sorted_input_target(pred_hazard, y)

        #identify patients alive at the time of each event in a dict
        at_risk_dict = self.make_at_risk_dict(np.array(sorted_y[0][:,0])) #key: time of an event; corresponding value: array of indexes of (sorted) patients who've had no event/censoring beforehand (including themselves)

        #define patients with event
        idx_pat_event = torch.where(sorted_y[0][:,1]==1.)[0]
        timepoint_case_patevent = np.where([np.array(timepoints_pred) == sorted_y[0][idx,0].item() for idx in idx_pat_event])[1] if(not GAINED_db) else \
            np.where([np.array(timepoints_pred) == round(sorted_y[0][idx,0].item(),4) for idx in idx_pat_event])[1]
            #vector of shape [n case patients (=n patients with event)]
        # print(idx_pat_event)
        # print(len(idx_pat_event))
        # print(timepoint_case_patevent)
        # print(timepoint_case_patevent.shape)

        #Get case and sample randomly n_control patients from these alive at the time of the event of case patient
        # g_case = torch.tensor([sorted_pred[0][idx_pat_event[i], timepoint_case_patevent[i]] for i in range(len(idx_pat_event))]) #= pred of each patient with event at the time of its event #Stops grad if computed this way
        idx_case=[[idx_pat_event[i].item(), timepoint_case_patevent[i]] for i in range(len(idx_pat_event))]
        g_case=sorted_pred[0][idx_case[0][0],idx_case[0][1]].unsqueeze(0)
        for i in range(len(idx_pat_event)-1):
            g_case = torch.concat((g_case,sorted_pred[0][idx_case[i+1][0],idx_case[i+1][1]].unsqueeze(0))) #= pred of each patient with event at the time of its event  #Awfully computed but with grad...


        control_idx = self.sample_alive_from_dates(np.array(sorted_y[0][idx_pat_event,0]), at_risk_dict, n_control) #np array of shape (number of g_case, n_control)
        # print(control_idx)
        # print(control_idx.shape)
        # g_control = tt.TupleTree(torch.tensor([[sorted_pred[0][control_idx[i,j],timepoint_case_patevent[i]] for i in range(len(idx_pat_event))] for j in range(len(control_idx[0]))])) #version working well but with no grad
        # print(g_control)
        idx_ctrl_formatted = [[[control_idx[i,j],timepoint_case_patevent[i]] for i in range(len(idx_pat_event))] for j in range(len(control_idx[0]))]

        controls = sorted_pred[0][idx_ctrl_formatted[0][0][0],idx_ctrl_formatted[0][0][1]].unsqueeze(0)
        for j in range(len(idx_ctrl_formatted[0])-1):
            controls = torch.concat((controls,sorted_pred[0][idx_ctrl_formatted[0][j+1][0],idx_ctrl_formatted[0][j+1][1]].unsqueeze(0)))
            #first control set
        if(len(idx_ctrl_formatted)>1):
            for i in range(len(idx_ctrl_formatted)-1):
                ctrl_x = sorted_pred[0][idx_ctrl_formatted[i+1][0][0],idx_ctrl_formatted[i+1][0][1]].unsqueeze(0)
                for j in range(len(idx_ctrl_formatted[0])-1):
                    ctrl_x = torch.concat((ctrl_x,sorted_pred[0][idx_ctrl_formatted[i+1][j+1][0],idx_ctrl_formatted[i+1][j+1][1]].unsqueeze(0)))
                ctrl_x = ctrl_x
                controls = tt.TupleTree((controls, ctrl_x)) 
                # print('g_control',i)
                # print(controls)
            g_control = controls
        else:
            g_control = tt.TupleTree(controls.unsqueeze(0)) 
        # print(g_control)
        # print('check selection g_control')
        # print(sorted_y)
        # print(sorted_pred)
        # print(control_idx)
        # print(g_control)

        # if(sorted_y[0][-1,1]==1.0): 
        #     print(sorted_y[0])
        #     print(at_risk_dict)
        #     print(idx_pat_event)
        #     print(control_idx)
        #     raise ValueError('check ce qu"il si passe qd le dernier patient a un événement: on le considère lui-mm dans les patients de ctrl? Comment on le gère?')

        return g_case, g_control
#g_case should be a tensor of shape (number of samples of patients WITH EVENT among the N sent to function), each value corresponding to the prediction for its time Ti
#g_control should be as well a tuple of shape (n_control), each element of n_control being a tensor of shape (number of samples ...), with
    # each value corresponding to the prediction of the patient 'control' for the time Ti of the corresponding case





# class CoxCCDataset(torch.utils.data.Dataset):
#     def __init__(self, input, durations, events, n_control=1):
#         df_train_target = pandas.DataFrame(dict(duration=durations, event=events))
#         self.durations = df_train_target.loc[lambda x: x['event'] == 1]['duration']
#         self.at_risk_dict = make_at_risk_dict(durations)

#         self.input = tt.tuplefy(input)
#         assert type(self.durations) is pd.Series
#         self.n_control = n_control

#     def __getitem__(self, index):
#         if (not hasattr(index, '__iter__')) and (type(index) is not slice):
#             index = [index]
#         fails = self.durations.iloc[index]
#         x_case = self.input.iloc[fails.index]
#         control_idx = sample_alive_from_dates(fails.values, self.at_risk_dict, self.n_control)
#         x_control = tt.TupleTree(self.input.iloc[idx] for idx in control_idx.transpose())
#         return tt.tuplefy(x_case, x_control).to_tensor()

#     def __len__(self):
#         return len(self.durations)


# class CoxTimeDataset(CoxCCDataset):
#     def __init__(self, input, durations, events, n_control=1):
#         super().__init__(input, durations, events, n_control)
#         self.durations_tensor = tt.tuplefy(self.durations.values.reshape(-1, 1)).to_tensor()

#     def __getitem__(self, index):
#         if not hasattr(index, '__iter__'):
#             index = [index]
#         durations = self.durations_tensor.iloc[index]
#         case, control = super().__getitem__(index)
#         case = case + durations
#         control = control.apply_nrec(lambda x: x + durations)
#         return tt.tuplefy(case, control)


#     make_dataset = pycox.models.data.CoxTimeDataset


#     @staticmethod
#     def _sorted_input_target(input, target):
#         input, target = tt.tuplefy(input, target).to_numpy()
#         durations, _ = target
#         idx_sort = np.argsort(durations)
#         if (idx_sort == np.arange(0, len(idx_sort))).all():
#             return input, target
#         input = tt.tuplefy(input).iloc[idx_sort]
#         target = tt.tuplefy(target).iloc[idx_sort]
#         return input, target

#     def make_dataloader(self, data, batch_size, shuffle=True, num_workers=0, n_control=1):
#         """Dataloader for training. Data is on the form (input, target), where
#         target is (durations, events).
        
#         Arguments:
#             data {tuple} -- Tuple containing (input, (durations, events)).
#             batch_size {int} -- Batch size.
        
#         Keyword Arguments:
#             shuffle {bool} -- If shuffle in dataloader (default: {True})
#             num_workers {int} -- Number of workers in dataloader. (default: {0})
#             n_control {int} -- Number of control samples in dataloader (default: {1})
        
#         Returns:
#             dataloader -- Dataloader for training.
#         """
#         input, target = self._sorted_input_target(*data)
#         durations, events = target

#         dataset = self.make_dataset(input, durations, events, n_control)
#         dataloader = tt.data.DataLoaderBatch(dataset, batch_size=batch_size,
#                                              shuffle=shuffle, num_workers=num_workers)
#         return dataloader







#     def compute_metrics(self, input, metrics):
#         if (self.loss is None) and (self.loss in metrics.values()):
#             raise RuntimeError(f"Need to specify a loss (self.loss). It's currently None")
#         input = self._to_device(input)
#         batch_size = input.lens().flatten().get_if_all_equal()
#         if batch_size is None:
#             raise RuntimeError("All elements in input does not have the same length.")
#         case, control = input # both are TupleTree
#         input_all = tt.TupleTree((case,) + control).cat()
#         g_all = self.net(*input_all)
#         g_all = tt.tuplefy(g_all).split(batch_size).flatten()
#         g_case = g_all[0]
#         g_control = g_all[1:]
#         #res = {name: metric(g_case, g_control) for name, metric in metrics.items()}
#         main_loss = self._criterion(g_case, g_control)


# CoxCCLoss:
#     def forward(self, g_case: Tensor, g_control: TupleTree) -> Tensor:
#         single = False
#         if hasattr(g_control, 'shape'):
#              if g_case.shape == g_control.shape:
#                 return cox_cc_loss_single_ctrl(g_case, g_control, self.shrink)
#         elif (len(g_control) == 1) and (g_control[0].shape == g_case.shape):
#                 return cox_cc_loss_single_ctrl(g_case, g_control[0], self.shrink)
#         return cox_cc_loss(g_case, g_control, self.shrink, self.clamp)
    








# a














# class CoxCCDataset(torch.utils.data.Dataset):
#     def __init__(self, input, durations, events, n_control=1):
#         df_train_target = pd.DataFrame(dict(duration=durations, event=events))
#         self.durations = df_train_target.loc[lambda x: x['event'] == 1]['duration']
#         self.at_risk_dict = make_at_risk_dict(durations)

#         self.input = tt.tuplefy(input)
#         assert type(self.durations) is pd.Series
#         self.n_control = n_control

#     def __getitem__(self, index):
#         if (not hasattr(index, '__iter__')) and (type(index) is not slice):
#             index = [index]
#         fails = self.durations.iloc[index]
#         x_case = self.input.iloc[fails.index]
#         control_idx = sample_alive_from_dates(fails.values, self.at_risk_dict, self.n_control)
#         x_control = tt.TupleTree(self.input.iloc[idx] for idx in control_idx.transpose())
#         return tt.tuplefy(x_case, x_control).to_tensor()

#     def __len__(self):
#         return len(self.durations)


# class CoxTimeDataset(CoxCCDataset):
#     def __init__(self, input, durations, events, n_control=1):
#         super().__init__(input, durations, events, n_control)
#         self.durations_tensor = tt.tuplefy(self.durations.values.reshape(-1, 1)).to_tensor()

#     def __getitem__(self, index):
#         if not hasattr(index, '__iter__'):
#             index = [index]
#         durations = self.durations_tensor.iloc[index]
#         case, control = super().__getitem__(index)
#         case = case + durations
#         control = control.apply_nrec(lambda x: x + durations)
#         return tt.tuplefy(case, control)


#     make_dataset = pycox.models.data.CoxTimeDataset


#     @staticmethod
#     def _sorted_input_target(input, target):
#         input, target = tt.tuplefy(input, target).to_numpy()
#         durations, _ = target
#         idx_sort = np.argsort(durations)
#         if (idx_sort == np.arange(0, len(idx_sort))).all():
#             return input, target
#         input = tt.tuplefy(input).iloc[idx_sort]
#         target = tt.tuplefy(target).iloc[idx_sort]
#         return input, target

#     def make_dataloader(self, data, batch_size, shuffle=True, num_workers=0, n_control=1):
#         """Dataloader for training. Data is on the form (input, target), where
#         target is (durations, events).
        
#         Arguments:
#             data {tuple} -- Tuple containing (input, (durations, events)).
#             batch_size {int} -- Batch size.
        
#         Keyword Arguments:
#             shuffle {bool} -- If shuffle in dataloader (default: {True})
#             num_workers {int} -- Number of workers in dataloader. (default: {0})
#             n_control {int} -- Number of control samples in dataloader (default: {1})
        
#         Returns:
#             dataloader -- Dataloader for training.
#         """
#         input, target = self._sorted_input_target(*data)
#         durations, events = target
#         dataset = self.make_dataset(input, durations, events, n_control)
#         dataloader = tt.data.DataLoaderBatch(dataset, batch_size=batch_size,
#                                              shuffle=shuffle, num_workers=num_workers)
#         return dataloader



#         if target is not None:
#             input = (input, target)
#         dataloader = self.make_dataloader(input, batch_size, shuffle, num_workers, **kwargs)
        
#         for _ in range(epochs):
#             if stop:
#                 break
#             stop = self.callbacks.on_epoch_start()
#             if stop:
#                 break
#             for data in dataloader:
#                 stop = self.callbacks.on_batch_start()
#                 if stop:
#                     break
#                 self.optimizer.zero_grad()
#                 self.batch_metrics = self.compute_metrics(data, self.metrics)
#                 self.batch_loss = self.batch_metrics["loss"]
















# class EPN(TorchCustomModel):
#     """
#     Error Passing Network model
#     """

#     def __init__(self,
#                  previous_pred_idx: int,
#                  pred_mu: float,
#                  pred_std: float,
#                  alpha: float = 0,
#                  beta: float = 0,
#                  num_cont_col: Optional[int] = None,
#                  cat_idx: Optional[List[int]] = None,
#                  cat_sizes: Optional[List[int]] = None,
#                  cat_emb_sizes: Optional[List[int]] = None,
#                  verbose: bool = False,
#                  similarity_kernel: str = SimilarityKernel.ATTENTION,
#                  n_neighbors: int = None):

#         # We call parent's constructor
#         super().__init__(criterion=MSELoss(),
#                          criterion_name='MSE',
#                          eval_metric=RootMeanSquaredError(),
#                          output_size=1,
#                          alpha=alpha,
#                          beta=beta,
#                          num_cont_col=num_cont_col,
#                          cat_idx=cat_idx,
#                          cat_sizes=cat_sizes,
#                          cat_emb_sizes=cat_emb_sizes,
#                          additional_input_args=None,
#                          verbose=verbose)

#         # Index indicating which column of the dataset is associated to previous
#         # predictions made by another model
#         self._prediction_idx = previous_pred_idx

#         # We set variables needed to set back predictions in their original scale
#         self._pred_mu = pred_mu
#         self._pred_std = pred_std

#         # Key and Query projection layers
#         self._key_projection = None
#         self._query_projection = None

#         # Scaling factor
#         self._dk = sqrt(Tensor([self._input_size]))

#         # Attention map cache
#         self._attn_cache = None

#         # Number of neighbors and similarity metric
#         self._n_neighbors = n_neighbors
#         self._similarity_kernel = similarity_kernel
#         self._compute_similarity = self._define_similarity_kernel(similarity_kernel)

    # @property
    # def attn_cache(self) -> Tensor:
    #     return self._attn_cache

    # def _define_similarity_kernel(self, similarity_kernel):
    #     if similarity_kernel == SimilarityKernel.ATTENTION:
    #         # Key and Query projection layers
    #         # We decrease the input size by one because one column contains predicted targets and will be removed
    #         self._key_projection = Linear(self._input_size - 1, self._input_size)
    #         self._query_projection = Linear(self._input_size - 1, self._input_size)
    #         # Use the whole dataset when using attention-based kernel similarity
    #         self._n_neighbors = None

    #         def compute_similarity(x1: Tensor, x2: Tensor) -> Tensor:
    #             return matmul(self._key_projection(x1), self._query_projection(x2).t()) / self._dk
    #     elif similarity_kernel == SimilarityKernel.DOT:
    #         def compute_similarity(x1: Tensor, x2: Tensor) -> Tensor:
    #             return matmul(x1, x2.t())
    #     elif similarity_kernel == SimilarityKernel.COSINE:
    #         def compute_similarity(x1: Tensor, x2: Tensor) -> Tensor:
    #             return matmul(x1 / torch.norm(x1, dim=1, keepdim=True),
    #                           (x2 / torch.norm(x2, dim=1, keepdim=True)).t())
    #     else:
    #         raise ValueError(f'Similarity kernel: {similarity_kernel} not implemented')

    #     return compute_similarity

####################################################################################################################################################
    # def _execute_valid_step(self,
    #                         valid_data: Tuple[Optional[DataLoader], PetaleDataset],
    #                         early_stopper: EarlyStopper) -> bool:
    #     """
    #     Executes an inference step on the validation data

    #     Args:
    #         valid_data: dataset

    #     Returns: True if we need to early stop
    #     """
    #     if valid_data[0] is None:
    #         return False

    #     # We extract the valid dataloader and the complete dataset
    #     valid_loader, dataset = valid_data

    #     # Set model for evaluation
    #     self.eval()
    #     epoch_loss, epoch_score = 0, 0

    #     # We extract the data of training and valid set
    #     x, y, all_idx = dataset[dataset.train_mask + dataset.valid_mask]

    #     # We execute one inference step on validation set
    #     with no_grad():

    #         for _, _, idx in valid_loader:
    #             # We perform the forward pass
    #             batch_pos_idx = [all_idx.index(i) for i in idx]
    #             output = self(x, y, batch_pos_idx)

    #             # We calculate the loss and the score
    #             epoch_loss += self.loss(output, y[batch_pos_idx]).item()
    #             epoch_score += self._eval_metric(output, y[batch_pos_idx])

    #     # We update evaluations history
    #     mean_epoch_score = self._update_evaluations_progress(epoch_loss, epoch_score,
    #                                                          nb_batch=len(valid_loader),
    #                                                          mask_type=MaskType.VALID)

    #     # We check early stopping status
    #     early_stopper(mean_epoch_score, self)

    #     if early_stopper.early_stop:
    #         return True

    #     return False

    # def _execute_train_step(self, train_data: Tuple[Optional[DataLoader], PetaleDataset]) -> float:
    #     """
    #     Executes one training epoch

    #     Args:
    #         train_data: dataset

    #     Returns: mean epoch loss
    #     """
    #     # We set the model for training
    #     self.train()
    #     epoch_loss, epoch_score = 0, 0

    #     # We extract the valid dataloader and the complete dataset
    #     train_loader, dataset = train_data

    #     # We extract the features related to all the train mask
    #     x, y, all_idx = dataset[dataset.train_mask]

    #     for _, _, idx in train_loader:
    #         # We clear the gradients
    #         self._optimizer.zero_grad()

    #         # We perform the weight update
    #         batch_pos_idx = [all_idx.index(i) for i in idx]
    #         pred, loss = self._update_weights([x, y, batch_pos_idx], y[batch_pos_idx])

    #         # We update the metrics history
    #         score = self._eval_metric(pred, y[batch_pos_idx])
    #         epoch_loss += loss
    #         epoch_score += score

    #     # We update evaluations history
    #     mean_epoch_loss = self._update_evaluations_progress(epoch_loss, epoch_score,
    #                                                         nb_batch=len(train_loader),
    #                                                         mask_type=MaskType.TRAIN)
    #     return mean_epoch_loss
####################################################################################################################################################

    # def forward(self,
    #             x: Tensor,
    #             y: Tensor,
    #             test_idx: Optional[List[int]] = None) -> Tensor:
    #     """
    #     Executes a forward pass.

    #     Args:
    #         x: (N, D) tensor with features
    #         y: (N, 1) tensor with targets
    #         test_idx: List of idx associated to test data points for which we want to calculate smooth targets.
    #                   If None, values are returned for all idx.

    #     Returns: (N, 1) tensor with smoothed targets
    #     """

    #     # We extract previous prediction made by another model
    #     y_hat = (x[:, self._prediction_idx] * self._pred_std) + self._pred_mu

    #     # We calculate the errors made on these predictions
    #     errors = y - y_hat
    #     # We initialize a list of tensors to concatenate
    #     new_x = []

    #     # We extract continuous data
    #     if len(self._cont_idx) != 0:
    #         new_x.append(x[:, [i for i in self._cont_idx if i != self._prediction_idx]])

    #     # We perform entity embeddings on categorical features
    #     if len(self._cat_idx) != 0:
    #         new_x.append(self._embedding_block(x))

    #     # We concatenate all inputs
    #     x = cat(new_x, 1)

    #     if test_idx is None:

    #         # We compute the scaled-dot product
    #         att = self._compute_similarity(x, x)

    #         # We set the diagonal to zero
    #         att = att * (1 - eye(att.shape[0], att.shape[1]))

    #         # Get the k neighbors
    #         if self._n_neighbors != None:
    #             att, indices = torch.topk(att, k=max(round(self._n_neighbors*att.shape[1]), 1), dim=-1)
    #             errors = errors[indices]

    #         else:
    #             # Reshape the dataset to len(x) * number of total nodes
    #             errors = errors.unsqueeze(1).repeat(1, len(x)).permute(1, 0)

    #         # We apply the softmax
    #         att = softmax(att, dim=-1)

    #     else:

    #         # We compute the scaled-dot product
    #         att = self._compute_similarity(x[test_idx, :], x)

    #         if not self.training:

    #             # We set the attention given to all test elements to zero
    #             # Therefore test elements cannot attend to errors of other test elements (including their own)
    #             att[:, test_idx] = 0

    #         else:

    #             # We only make sure that self attention value of test elements are zeroed
    #             att[range(len(test_idx)), test_idx] = 0

    #         # Get the k neighbors
    #         if self._n_neighbors != None:
    #             att, indices = torch.topk(att, k=max(round(self._n_neighbors*att.shape[1]), 1), dim=1)
    #             errors = errors[indices]

    #         else:
    #             # Reshape the dataset to test_idx * number of total nodes
    #             errors = errors.unsqueeze(1).repeat(1, len(test_idx)).permute(1, 0)

    #         # We apply the softmax
    #         self._attn_cache = att = softmax(att, dim=-1)

    #         # We only keep predictions previously made for test idx
    #         y_hat = y_hat[test_idx]
    #     return (matmul(att, errors.t())[:, 0] + y_hat).squeeze(dim=-1)

    # def predict(self,
    #             dataset: PetaleDataset,
    #             mask: Optional[List[int]] = None) -> Tensor:
    #     """
    #     Returns the real-valued predictions for all samples
    #     in a particular set (default = test)

    #     Args:
    #         dataset: PetaleDataset which its items are tuples (x, y, idx) where
    #                  - x : (N,D) tensor with D-dimensional samples
    #                  - y : (N,) tensor with classification labels
    #                  - idx : (N,) tensor with idx of samples according to the whole dataset
    #         mask: list of dataset idx for which we want to predict target

    #     Returns: (N,) tensor
    #     """

    #     # Set mask value if not provided
    #     if mask is None:
    #         mask = dataset.test_mask

    #     # Set model for evaluation
    #     self.eval()

    #     # We look for the idx that are not in the training set
    #     added_idx = [i for i in mask if i not in dataset.train_mask]

    #     # Execute a forward pass and apply a softmax
    #     with no_grad():
    #         x, y, idx = dataset[dataset.train_mask + added_idx]
    #         if len(added_idx) > 0:
    #             return self(x, y, [idx.index(i) for i in added_idx])
    #         else:
    #             return self(x, y)


# class PetaleEPN(TorchRegressorWrapper):
#     """
#     Error Passing Network wrapper for the Petale framework
#     """

#     def __init__(self,
#                  previous_pred_idx: int,
#                  pred_mu: float,
#                  pred_std: float,
#                  alpha: float = 0,
#                  beta: float = 0,
#                  num_cont_col: Optional[int] = None,
#                  cat_idx: Optional[List[int]] = None,
#                  cat_sizes: Optional[List[int]] = None,
#                  cat_emb_sizes: Optional[List[int]] = None,
#                  lr: float = 0.05,
#                  rho: float = 0,
#                  batch_size: Optional[int] = None,
#                  max_epochs: int = 200,
#                  patience: int = 15,
#                  verbose: bool = False,
#                  similarity_kernel: str = SimilarityKernel.ATTENTION,
#                  n_neighbors: int = None):
#         # Creation of the model
#         model = EPN(previous_pred_idx=previous_pred_idx,
#                     pred_mu=pred_mu,
#                     pred_std=pred_std,
#                     alpha=alpha,
#                     beta=beta,
#                     num_cont_col=num_cont_col,
#                     cat_idx=cat_idx,
#                     cat_sizes=cat_sizes,
#                     cat_emb_sizes=cat_emb_sizes,
#                     verbose=verbose,
#                     similarity_kernel=similarity_kernel,
#                     n_neighbors=n_neighbors)

#         # Call of parent's constructor
#         # Valid batch size is set to None to use full batch at once
#         super().__init__(model=model,
#                          train_params=dict(lr=lr,
#                                            rho=rho,
#                                            batch_size=batch_size,
#                                            valid_batch_size=None,
#                                            patience=patience,
#                                            max_epochs=max_epochs,
#                                            include_dataset=True))

#     @staticmethod
#     def get_hps() -> List[HP]:
#         """
#         Returns a list with the hyperparameters associated to the model

#         Returns: list of hyperparameters
#         """
#         return list(EPNHP())


# class EPNHP:
#     """
#     EPN hyperparameters
#     """
#     ALPHA = NumericalContinuousHP("alpha")
#     BATCH_SIZE = NumericalIntHP("batch_size")
#     BETA = NumericalContinuousHP("beta")
#     LR = NumericalContinuousHP("lr")
#     RHO = NumericalContinuousHP("rho")
#     NEIGHBORS = NumericalIntHP("n_neighbors")

#     def __iter__(self):
#         return iter([self.ALPHA, self.BATCH_SIZE, self.BETA, self.LR, self.RHO, self.NEIGHBORS])
