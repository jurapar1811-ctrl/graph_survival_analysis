import os
import sys
import pandas
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
sys.path.insert(1, './') 
from functions.miscellaneous import get_clinical



def unpickle(file):
    objs = []
    while 1:
        try:
            objs.append(pickle.load(file))
        except EOFError:
            break
    return objs



def getting_feat_new_DB(parameters, get_clin=True, get_img_feat=True, get_graphs=False, repo='', cluster=False):
    output_feature, clin_not_stand = [],[]
    path_DB = repo+parameters.DB if(cluster) else parameters.DB
    list_pat = [np.int64(parameters.patients_to_use[i]) for i in range(len(parameters.patients_to_use))]
    if(parameters.scaler == 'stand_scaler'):
        scaler = StandardScaler()
    elif(parameters.scaler == 'rob_scaler'):
        scaler = RobustScaler()
    else:
        raise ValueError("scaler is not in ['stand_scaler', 'rob_scaler'] (in parameters.py)")

    if get_clin:    
        df = pandas.read_csv(path_DB, encoding='unicode_escape')
        df = df[df["patients_id"].isin(list_pat)]
        if(not parameters.largest): df = df[df["lesion_id"]==1] 
        if(not parameters.clin_onehot):
            clinical = np.transpose(np.vstack((df['clin_age'], df['clin_Ann_Arbor_stage'], df['clin_ECOG_scale'], df['clin_n_extranodal_site'], df['clin_LDH'],\
                                df['clin_LDH_categorical'], df['clin_aaIPI'], df['clin_chemoterapy_regimen'], df['clin_autologous_cell_transplant'],\
                                df['clin_salvage_therapy'])))
        else:
            clinical = np.transpose(np.vstack((df['clin_age'], df['clin_Ann_Arbor_stage_1'], df['clin_Ann_Arbor_stage_2'], df['clin_Ann_Arbor_stage_3'], df['clin_Ann_Arbor_stage_4'],\
                                    df['clin_ECOG_scale_0'], df['clin_ECOG_scale_1'], df['clin_ECOG_scale_2'], df['clin_ECOG_scale_3'], df['clin_n_extranodal_site'], df['clin_LDH'],\
                                    df['clin_LDH_categorical'], df['clin_aaIPI_1'], df['clin_aaIPI_2'], df['clin_aaIPI_3'], df['clin_chemoterapy_regimen'],\
                                    df['clin_autologous_cell_transplant'], df['clin_salvage_therapy'])))
            
        #Features order: age-annarbor-ecog-extranodal-ldh-ldhcat-aaIPI-chimio-cell-salvage
        output_feature.append(scaler.fit_transform(clinical))
        output_feature = output_feature[0]
        clin_not_stand.append(clinical)
    
    if get_img_feat:
        df = pandas.read_csv(path_DB, encoding='unicode_escape')
        df = df[df["patients_id"].isin(list_pat)]

        #We take the columns corresponding to each type of data:
        list_les_centers = np.array(list(df))[[list(df)[i].find('lesion_center')!=-1 for i in range(len(list(df)))]]
        list_class = np.array(list(df))[[list(df)[i].find('class')!=-1 for i in range(len(list(df)))]]
        list_rad = np.array(list(df))[[list(df)[i].find('rad')!=-1 for i in range(len(list(df)))]]
        
        #We make an array with the features of each lesion:
        list_centers_array = np.transpose(np.vstack([df[list_les_centers[i]] for i in range (len(list_les_centers))]))
        classical_array = scaler.fit_transform(np.transpose(np.vstack([df[list_class[i]] for i in range (len(list_class))])))
        radiomics_array = scaler.fit_transform(np.transpose(np.vstack([df[list_rad[i]] for i in range (len(list_rad))])))

        #We make a list grouping the lesions of each patient:
        if(not parameters.largest):
            lesion_ids = np.array(df['lesion_id'])
            list_centers, classical, radiomics = [],[],[]
            last_pat = 0
            for i in range(len(lesion_ids)):
                if(i+1==len(lesion_ids)):
                    list_centers.append(list_centers_array[last_pat:i+1])
                    classical.append(classical_array[last_pat:i+1])
                    radiomics.append(radiomics_array[last_pat:i+1])
                elif(lesion_ids[i+1]==1):
                    list_centers.append(list_centers_array[last_pat:i+1])
                    classical.append(classical_array[last_pat:i+1])
                    radiomics.append(radiomics_array[last_pat:i+1])
                    last_pat = i+1
        else:
            list_centers = list(list_centers_array)
            classical = list(classical_array)
            radiomics = list(radiomics_array)

        if(not parameters.largest):
            for i in range(len(list_centers)):  #Standardized by patient
                list_centers[i] = scaler.fit_transform(list_centers[i])

        output_feature.append(list_centers)
        output_feature.append(classical)
        output_feature.append(radiomics)
    
    if get_graphs:
        mask_path = '/media/user/DD_These/These/DLBCL_new_masks/mask_suv3.0'
        imgfeat_patient_list = np.delete(np.array(os.listdir(mask_path)), np.where(np.array(os.listdir(mask_path)) == '51011101261002'))
        imgfeat_patient_list_sort = sorted(imgfeat_patient_list) #because the files used to extract the features were not sorted (cf os.listdir(mask_path))
        with open('./files_to_load/new_graphs.pickle', 'rb') as f:
            graphs_tmp = unpickle(f)
        with open('./files_to_load/new_dmaxs.pickle', 'rb') as f:
            dmax_tmp = unpickle(f)
        graphs, dmax = [],[]
        for i in range(len(imgfeat_patient_list_sort)):
            if imgfeat_patient_list_sort[i] in parameters.patients_to_use:
                graphs.append(graphs_tmp[np.where(imgfeat_patient_list == imgfeat_patient_list_sort[i])[0][0]])
                dmax.append(dmax_tmp[np.where(imgfeat_patient_list == imgfeat_patient_list_sort[i])[0][0]])

        #Function get_graph():
        #graph_array = get_graph(graphs, dmax) #graph features
        graph_array = np.zeros((len(graphs), 3))
        for i in range(len(graphs)):
            g = graphs[i]
            graph_array[i] = [g.num_nodes, g.num_edges, dmax[i]]

        np.save(parameters.PIK_dump_g_features, graph_array)
        
        output_feature.append(graphs)
        output_feature.append(graph_array)

    return output_feature, clin_not_stand[0]