import sys
import numpy as np
sys.path.insert(1, './') 
from functions.miscellaneous import get_labels


_, patient_list, _ = get_labels("./files_to_load/clinical_data_05_02.csv")


def test_split(feat, lab, patients_to_use):
    print("in")
    current_split_test_tmp = np.load('./files_to_load/tmp/tests_583.npy')
    current_split_test = patient_list[current_split_test_tmp] #ids of the patients in the test set

    # patients = [np.where(current_split_test==patients_to_use[i])[0][0] for i in range(len(patients_to_use)) if (int(patients_to_use[i]) in current_split_test)]
    patients = [np.where(np.array(patients_to_use)==str(current_split_test[i]))[0][0] for i in range(len(current_split_test)) if (str(current_split_test[i]) in patients_to_use)]
    out_feat1, out_lab1, out_feat2, out_lab2 = [],[],[],[]
    for i in range(len(patients)):
        out_feat1.append(feat[patients[i]])
        out_lab1.append(lab[patients[i]])
    not_patient = np.delete(range(len(patients_to_use)), patients)
    for i in range(len(not_patient)):
        out_feat2.append(feat[not_patient[i]])
        out_lab2.append(lab[not_patient[i]])

    return np.array(out_feat1), np.array(out_feat2), np.array(out_lab1), np.array(out_lab2)
#feat[patients], lab[patients], feat[np.delete(range(len(current_split_test)), patients)], lab[np.delete(range(len(current_split_test)), patients)]



def train_val_split(feat, lab, patients_to_use, iter):
    current_split_train_tmp = np.load('./files_to_load/tmp/trains_583.npy')
    current_split_val_tmp = np.load('./files_to_load/tmp/vals_583.npy')
    current_split_train = patient_list[current_split_train_tmp]
    current_split_val = patient_list[current_split_val_tmp]

    patients_train = [np.where(np.array(patients_to_use)==str(current_split_train[iter][i]))[0][0] for i in range(len(current_split_train[iter])) if (str(current_split_train[iter][i]) in patients_to_use)]
    patients_val = [np.where(np.array(patients_to_use)==str(current_split_val[iter][i]))[0][0] for i in range(len(current_split_val[iter])) if (str(current_split_val[iter][i]) in patients_to_use)]

    #/!\: there we do not have all patients, but just those we are not in the test set
    current_split_test_tmp = np.load('./files_to_load/tmp/tests_583.npy')
    current_split_test = patient_list[current_split_test_tmp]
    patients_test = [np.where(np.array(patients_to_use)==str(current_split_test[i]))[0][0] for i in range(len(current_split_test)) if (str(current_split_test[i]) in patients_to_use)]
    table = np.delete(range(len(patients_to_use)), patients_test)


    out_feat1, out_lab1, out_feat2, out_lab2 = [],[],[],[]
    for i in range(len(patients_val)):
        out_feat1.append(feat[np.where(table==patients_val[i])[0][0]])
        out_lab1.append(lab[np.where(table==patients_val[i])[0][0]])
    for i in range(len(patients_train)):
        out_feat2.append(feat[np.where(table==patients_train[i])[0][0]])
        out_lab2.append(lab[np.where(table==patients_train[i])[0][0]])
    for i in range(len(table)):    
        if(table[i] not in patients_val and table[i] not in patients_train):
            out_feat2.append(feat[i])
            out_lab2.append(lab[i])
    
    return np.array(out_feat2), np.array(out_feat1), np.array(out_lab2), np.array(out_lab1)

