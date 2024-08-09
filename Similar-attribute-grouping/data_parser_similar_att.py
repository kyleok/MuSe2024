import os
from typing import List, Dict, Optional, Union

import numpy as np
import pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from data_parser import get_data_partition, fit_normalizer, load_humor_subject, load_perception_subject

from config import HUMOR, PERCEPTION


def load_sim_att_data(task:str,
              paths:Dict[str, str],
              feature:str,
              label_dims: Optional[str],
              normalize: Optional[Union[bool, StandardScaler]] = True,
              save=False,
              ids: Optional[Dict[str, List[str]]] = None,
              data_file_suffix: Optional[str]=None) \
        -> Dict[str, Dict[str, List[np.ndarray]]]:
    """
    Loads the complete data sets
    :param task: task
    :param paths: dict for paths to data and partition file
    :param feature: feature to load
    :param label_dims: label dimensions to load labels for - only relevant for perception task
    :param normalize: whether normalization is desired
    :param save: whether to cache the loaded data as .pickle
    :param segment_train: whether to do segmentation on the training data
    :param ids: only consider these IDs (map 'train', 'devel', 'test' to list of ids) - only relevant for personalisation
    :param data_file_suffix: optional suffix for data file, may be useful for personalisation
    :return: dict with keys 'train', 'devel' and 'test', each in turn a dict with keys:
        feature: list of ndarrays shaped (seq_length, features)
        labels: corresponding list of ndarrays shaped (seq_length, 1) for n-to-n tasks like stress, (1,) for n-to-1
            task humor, (4,) for n-to-4 task mimic
        meta: corresponding list of ndarrays shaped (seq_length, metadata_dim) where seq_length=1 for n-to-1/n-to-4
    """

    is_loaded = False
    data = {'train': {'feature': [], 'label': {label_dim: [] for label_dim in label_dims}, 'meta': {label_dim: [] for label_dim in label_dims}},
        'devel': {'feature': [], 'label': {label_dim: [] for label_dim in label_dims}, 'meta': {label_dim: [] for label_dim in label_dims}},
        'test': {'feature': [], 'label': {label_dim: [] for label_dim in label_dims}, 'meta': {label_dim: [] for label_dim in label_dims}}}
    
    for label_dim in label_dims:
        data_file_name = f'data_{task}_{feature}_{label_dim + "_" if len(label_dim) > 0 else ""}_{"norm_" if normalize else ""}' \
                        f'{f"_{data_file_suffix}" if data_file_suffix else ""}.pkl'
        data_file = os.path.join(paths['data'], data_file_name)

        print(f'Constructing {label_dim} data from scratch ...')
        subject2partition, partition2subject = get_data_partition(paths['partition'])

        if not(normalize is None):
            if type(normalize) == bool:
                    normalizer = fit_normalizer(task=task, feature=feature) if normalize else None
            else:
                normalizer = normalize
        else:
            normalizer = None
        for partition, subject_ids in partition2subject.items():
            if ids:
                subject_ids = [s for s in subject_ids if s in ids[partition]]

            for subject_id in tqdm(subject_ids):
                if task == HUMOR:
                    features, labels, metas = load_humor_subject(feature=feature, subject_id=subject_id,
                                                                normalizer=normalizer)
                elif task == PERCEPTION:
                    features, labels, metas = load_perception_subject(feature=feature, subject_id=subject_id,
                                                                    normalizer=normalizer, label_dim=label_dim)
                if not is_loaded:
                    data[partition]['feature'].extend(features)

                data[partition]['meta'][label_dim].extend(metas)
                data[partition]['label'][label_dim].extend(labels)

        is_loaded = True

        if save:  # save loaded and preprocessed data
            print('Saving data...')
            pickle.dump(data, open(data_file, 'wb'))
    return data