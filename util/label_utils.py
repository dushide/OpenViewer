# coding=utf-8
import os
import numpy as np
from sklearn.model_selection import train_test_split

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

def generate_partition(labels, indices,ratio):
    each_class_num = count_each_class_num(labels)
    labeled_each_class_num = {}  ## number of labeled samples for each class
    total_num = round(ratio * len(labels))
    for label in each_class_num.keys():
        labeled_each_class_num[label] = max(round(each_class_num[label] * ratio), 1)  # min is 1

    # index of labeled and unlabeled samples
    p_labeled = []
    p_unlabeled = []
    for idx in range(len(labels)):
        if (labeled_each_class_num[labels[idx]] > 0):
            labeled_each_class_num[labels[idx]] -= 1
            p_labeled.append(indices[idx])
            total_num -= 1
        else:
            p_unlabeled.append(indices[idx])
    return p_labeled, p_unlabeled

def reassign_labels(y, seen_labels, unseen_label_index):

    if isinstance(y, list):
        y = np.array(y)

    old_new_label_dict = {old_label:new_label for new_label, old_label in enumerate(seen_labels)}

    def convert_label(old_label):
        return old_new_label_dict[old_label] if old_label in old_new_label_dict else unseen_label_index

    new_y = [
        convert_label(label) for label in y
    ]

    new_y = np.array(new_y)

    return new_y


def special_train_test_split(y, unseen_label_index, test_size):

    if isinstance(y, list):
        y = np.array(y)

    seen_indices = np.where(y != unseen_label_index)[0]
    unseen_indices = np.where(y == unseen_label_index)[0]

    seen_train_indices, seen_test_indices =generate_partition(y[seen_indices],seen_indices,1 - test_size)

    train_indices = seen_train_indices
    test_indices = np.concatenate([seen_test_indices, unseen_indices], axis=0)
    return train_indices, test_indices


# y = [0, 1, 2, 3, 4,1,2, 3, 2,1, 3, 4,2, 1,2, 4, 3, 4, 5,2, 3, 9, 2]
# y = np.array(y)
# seen_labels = [2, 3, 9]
# new_y = reassign_labels(y, seen_labels, -1)
# train_indices, test_indices = special_train_test_split(new_y, unseen_label_index=-1, test_size=0.2)
# print(new_y)
# print(y[train_indices])
# print(new_y[train_indices])
# print("========")
# print(y[test_indices])
# print(new_y[test_indices])
