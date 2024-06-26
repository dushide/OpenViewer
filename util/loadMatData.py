import os

import h5py
import scipy.sparse as sp
import torch
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()  # <class 'numpy.ndarray'> (n_edges, 2)
    values = sparse_mx.data  # <class 'numpy.ndarray'> (n_edges,)
    shape = sparse_mx.shape  # <class 'tuple'>  (n_samples, n_samples)
    return coords, values, shape
def construct_sparse_float_tensor(np_matrix):
    """
    construct a sparse float tensor according a numpy matrix
    :param np_matrix: <class 'numpy.ndarray'>
    :return: torch.sparse.FloatTensor
    """
    sp_matrix = sp.csc_matrix(np_matrix)
    three_tuple = sparse_to_tuple(sp_matrix)
    sparse_tensor = torch.sparse.FloatTensor(torch.LongTensor(three_tuple[0].T),
                                             torch.FloatTensor(three_tuple[1]),
                                             torch.Size(three_tuple[2]))
    return sparse_tensor

def count_each_class_num(labels):
    '''
        Count the number of samples in each class
    '''
    count_dict = {}
    for label in labels:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict


def generate_partition(labels, ratio):
    each_class_num = count_each_class_num(labels)
    labeled_each_class_num = {}  ## number of labeled samples for each class
    total_num = round(ratio * len(labels))
    for label in each_class_num.keys():
        labeled_each_class_num[label] = max(round(each_class_num[label] * ratio), 1)  # min is 1

    # index of labeled and unlabeled samples
    p_labeled = []
    p_unlabeled = []
    for idx, label in enumerate(labels):
        if (labeled_each_class_num[label] > 0):
            labeled_each_class_num[label] -= 1
            p_labeled.append(idx)
            total_num -= 1
        else:
            p_unlabeled.append(idx)
    return p_labeled, p_unlabeled


def load_data(dataset, path="./data/", ):
    feature_list = []
    if dataset == "AwA" or dataset=="Reuters":
        data = h5py.File(path + dataset + '.mat')
        features = data['X']
        for i in range(features.shape[1]):
                feature_list.append(normalize(data[features[0][i]][:].transpose()))
        labels = data['Y'][:].flatten()
    else:
        data = sio.loadmat(path + dataset + '.mat')
        features = data['X']
        for i in range(features.shape[1]):
            features[0][i] = normalize(features[0][i])
            feature = features[0][i]
            if ss.isspmatrix_csr(feature):
                feature = feature.todense()
                print("sparse")
            # feature = torch.from_numpy(feature).float().to(args.device)
            feature_list.append(feature)
        labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    return feature_list, labels


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalization(data):
    maxVal = torch.max(data)
    minVal = torch.min(data)
    data = (data - minVal)//(maxVal - minVal)
    return data


