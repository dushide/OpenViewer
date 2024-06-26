import pickle
from torch.utils.data import Dataset

import numpy as np


class Multi_view_data(Dataset):
    """
    load multi-view data
    """

    def __init__(self,view_number,idx,feature_list, labels):
        """
        :param root: data name and path
        :param train: load training set or test set
        """
        super(Multi_view_data, self).__init__()

        self.x = dict()
        for v_num in range(view_number):
            self.x[v_num] = feature_list[v_num][[idx],:].squeeze()
        self.y =labels[idx]

    def __getitem__(self, index):
        data = dict()
        for v_num in range(len(self.x)):
            data[v_num] = (self.x[v_num][index]).astype(np.float32)
        target = self.y[index]

        return {
            'x':  data,
            'y': target,
            'index': index
        }

    def __len__(self):
        return len(self.y)

class MultiViewDataset(Dataset):
    def __init__(self, data_path='dataset/handwritten_6views_train.pkl'):
        super().__init__()
        self.x, self.y = pickle.load(open(data_path, 'rb'))

    def __getitem__(self, index):
        x = dict()
        for v in self.x.keys():
            x[v] = self.x[v][index]
        return {
            'x': x,
            'y': self.y[index],
            'index': index
        }

    def __len__(self):
        return len(self.y)
#
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
def generate_partition(labels,ind, ratio=0.1):
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
            p_labeled.append(ind[idx])
            total_num -= 1
        else:
            p_unlabeled.append(ind[idx])
    return p_labeled, p_unlabeled