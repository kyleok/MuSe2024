import os
from typing import List, Dict, Optional, Union

import numpy as np
import pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import os
from typing import List, Dict, Optional, Union, Tuple

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from config import PATH_TO_LABELS, PERCEPTION

from data_parser import get_data_partition


def load_feature_subject(feature_path, subject_id, normalizer, label_dim) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Loads extracted feature for a single subject for the perception task
    :param feature: feature name
    :param subject_id: subject name/ID
    :param normalizer: fitted StandardScaler, can be None if no normalization is desired. It is created in the load_data
    method, so no need to take care of that. It just needs to be called in the load_mimic_subject method somewhere
    to normalize the features
    :return: features, labels, metas.
        Assuming every subject consists of n segments of lengths l_1,...,l_n:
            features is a list (length n) of ndarrays of shape (l_i, feature_dim)  - each item corresponding to a segment
            labels is a ndarray of shape (n, num_classes) (labels for each element in the features list, assuming every segment
                has num_classes labels)
            metas is a ndarray of shape (n, 1, x) where x is the number of columns needed to describe the segment
                Typically something like (subject_id, segment_id, seq_start, seq_end) or the like
                They are only used to write the predictions: a prediction line consists of all the meta data associated
                    with one data point + the predicted label(s)
    """
    # parse label

    label_path = PATH_TO_LABELS[PERCEPTION]
    labels_df = pd.read_csv(label_path)
    label_values = labels_df[(labels_df.subj_id==int(subject_id))][label_dim].values
    assert len(label_values) == 1
    label = np.array([[label_values[0]]])

    feature_df = pd.read_csv(feature_path)
    feature_df = feature_df[feature_df['subj_id']==int(subject_id)]
    feature_df = feature_df.drop(columns=['subj_id'])

    features = feature_df.values
    if not (normalizer is None):
        features = normalizer.transform(features)
    features = [features]

    metas = np.array([subject_id]).reshape((1, 1, 1))

    return features, label, metas

def load_feature_fusion_data(task:str,
              paths:Dict[str, str],
              pt_model_names:Dict[str, str],
              label_dim: Optional[str],
              normalize: Optional[Union[bool, StandardScaler]] = True,
              save=False,
              ids: Optional[Dict[str, List[str]]] = None,
              data_file_suffix: Optional[str]=None) \
        -> Dict[str, Dict[str, List[np.ndarray]]]:
    """
    Loads the complete data sets
    :param task: task
    :param paths: dict for paths to data and partition file
    :param pt_model_names: dict for experiment names to get pretrained models
    :param label_dims: label dimensions to load labels for - only relevant for perception task
    :param normalize: whether normalization is desired
    :param save: whether to cache the loaded data as .pickle
    :param ids: only consider these IDs (map 'train', 'devel', 'test' to list of ids) - only relevant for personalisation
    :param data_file_suffix: optional suffix for data file, may be useful for personalisation
    :return: dict with keys 'train', 'devel' and 'test', each in turn a dict with keys:
        feature: list of ndarrays shaped (seq_length, features)
        labels: corresponding list of ndarrays shaped (seq_length, 1) for n-to-n tasks like stress, (1,) for n-to-1
            task humor, (4,) for n-to-4 task mimic
        meta: corresponding list of ndarrays shaped (seq_length, metadata_dim) where seq_length=1 for n-to-1/n-to-4
    """

    data = {'train': {'feature': [], 'label': [], 'meta': []},
            'devel': {'feature': [], 'label': [], 'meta': []},
            'test': {'feature': [], 'label': [], 'meta': []}}

    print('Constructing data from scratch ...')
    subject2partition, partition2subject = get_data_partition(paths['partition'])
    normalizer = None

    for partition, subject_ids in partition2subject.items():
        if ids:
            subject_ids = [s for s in subject_ids if s in ids[partition]]

        for subject_id in tqdm(subject_ids):
            for times, feature_type in enumerate(pt_model_names.keys()):
                feature_path = os.path.join(paths['data'], f'{pt_model_names[feature_type]}.csv')


                if task == PERCEPTION:
                    features, labels, metas = load_feature_subject(feature_path=feature_path, subject_id=subject_id,
                                                                    normalizer=normalizer, label_dim=label_dim)
        
                if times == 0:
                    data[partition]['feature'].extend(features)
                    data[partition]['label'].extend(labels)
                    data[partition]['meta'].extend(metas)
                else:
                    data[partition]['feature'][-1] = np.concatenate((data[partition]['feature'][-1], features[0]), axis=1)

    return data