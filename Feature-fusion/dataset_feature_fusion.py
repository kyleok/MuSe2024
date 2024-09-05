import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset
import numpy as np


class MuSeFeatureDataset(Dataset):
    def __init__(self, data, partition):
        super(MuSeFeatureDataset, self).__init__()
        self.partition = partition
        features, labels = data[partition]['feature'], data[partition]['label']
        metas = data[partition]['meta']
        self.feature_dim = features[0].shape[-1]
        self.n_samples = len(features)

        self.features = torch.tensor(np.array(features), dtype=torch.float)
        self.labels = torch.tensor(np.array(labels), dtype=torch.float)
        self.metas = metas
        pass

    def get_feature_dim(self):
        return self.feature_dim

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """

        :param idx:
        :return: feature, label, meta with
            feature: tensor of shape seq_len, feature_dim
            label: tensor of corresponding label(s) (shape 1 for n-to-1, else (seq_len,1))
            meta: list of lists containing corresponding meta data
        """
        feature = self.features[idx]
        label = self.labels[idx]
        meta = self.metas[idx]

        sample = feature, label, meta
        return sample


def custom_feature_collate_fn(data):
    """
    Custom collate function to ensure that the meta data are not treated with standard collate, but kept as ndarrays
    :param data:
    :return:
    """
    tensors = [d[:2] for d in data]
    np_arrs = [d[2] for d in data]
    coll_features, coll_labels = default_collate(tensors)
    np_arrs_coll = np.row_stack(np_arrs)
    return coll_features, coll_labels, np_arrs_coll