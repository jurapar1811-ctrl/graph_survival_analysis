import torch
from torch.utils.data import Dataset

class MilDataset(Dataset):
    '''
    Subclass of torch.utils.data.Dataset.
    Args:
      data:
      ids:
      labels:
      normalize:
    '''

    def __init__(self, data, ids, labels, normalize=False):
        self.data = data
        self.labels = labels
        self.ids = ids

        # Modify shape of bagids if only 1d tensor
        if (len(ids.shape) == 1):
            ids.resize_(1, len(ids))

        self.bags = torch.unique(self.ids[0])  #retrieve non-redundant patients_id 

        # Normalize
        if normalize:
            std = self.data.std(dim=0)
            mean = self.data.mean(dim=0)
            self.data = (self.data - mean) / std

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        data = self.data[self.ids[0] == self.bags[index]]
        bagids = self.ids[:, self.ids[0] == self.bags[index]]  # returns a ~vector [[patient_id{*nbROI_ds_pat}]]
        labels = self.labels[index]

        return data, bagids, labels

    def n_features(self):
        return self.data.size(1)


def collate(batch):
    '''
    '''
    batch_data = []
    batch_bagids = []
    batch_labels = []

    for sample in batch:
        batch_data.append(sample[0])
        batch_bagids.append(sample[1])
        batch_labels.append(sample[2])

    out_data = torch.cat(batch_data, dim=0)
    out_bagids = torch.cat(batch_bagids, dim=1)
    out_labels = torch.stack(batch_labels)

    return out_data, out_bagids, out_labels


def collate_np(batch):
    '''
        prepNN = torch.nn.Sequential(
            torch.nn.Linear(len(dataset.data[0]), 32),
            torch.nn.ReLU()
        )

        afterNN = torch.nn.Sequential(
            torch.nn.Linear(32, 1)
        )

        # Define model ,loss function and optimizer
        model1 = MILModel(prepNN, afterNN, torch.max)
    '''
    batch_data = []
    batch_bagids = []
    batch_labels = []

    for sample in batch:
        batch_data.append(sample[0])
        batch_bagids.append(sample[1])
        batch_labels.append(sample[2])

    out_data = torch.cat(batch_data, dim=0)
    out_bagids = torch.cat(batch_bagids, dim=1)
    out_labels = torch.tensor(batch_labels)

    return out_data, out_bagids, out_labels
