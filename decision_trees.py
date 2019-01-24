import numpy as np
import sys

np.set_printoptions(threshold=float('inf'))

clean_dataset = np.loadtxt('wifi_db/clean_dataset.txt')
noisy_dataset = np.loadtxt('wifi_db/noisy_dataset.txt')

# Find the best split to perform
def find_split(dataset):
    max_gain = float('-inf')
    split_point = None

    class_index = len(dataset[0]) - 1

    # TODO: Make floats

    # find gain for each attribute
    for i in range(len(dataset[0]) - 1):
        # sort whole dataset by current attribute (i)
        sorted_dataset = dataset[dataset[:,i].argsort()]

        split_points = find_split_points(sorted_dataset, i)
        gain = find_best_gain(split_points, sorted_dataset, i)

        if gain[0] > max_gain:
            max_gain = gain[0]
            split_point = (gain[1], i)

    return split_point


# Find all possible split points for the given column
def find_split_points(sorted_dataset, attribute):
    split_points = np.array([])

    class_index = len(sorted_dataset[0]) - 1

    for i in range(len(sorted_dataset) - 1):
        val = sorted_dataset[i][attribute]
        classification = sorted_dataset[i][class_index]
        val_next = sorted_dataset[i + 1][attribute]
        classification_next = sorted_dataset[i + 1][class_index]

        if val != val_next:
            mid = (val + val_next) / 2
            split_points = np.append(split_points, mid)

    return split_points

# Finds the largest gain given some split points and a sorted column
def find_best_gain(split_points, sorted_dataset, attribute):
    max_gain = float('-inf')
    gain_point = (float('-inf'), None)

    for split_point in split_points:
        left = sorted_dataset[sorted_dataset[:,attribute] < split_point]
        right = sorted_dataset[sorted_dataset[:,attribute] > split_point]
        curr_gain = gain(sorted_dataset, left, right)

        if curr_gain > max_gain:
            max_gain = curr_gain
            gain_point = (curr_gain, split_point)

    return gain_point

# Calculate entropy of the dataset
def entropy(dataset):
    class_index = len(dataset[0]) - 1
    classes = dataset[:,class_index]

    H = 0

    for i in range(1, 4):
        no_in_class = (classes == i).sum()

        if len(classes) == 0 or no_in_class == 0:
            continue

        pk = no_in_class / len(classes)
        H = H - pk * np.log2(pk)

    return H

def remainder(left, right):
    num_left = left.shape[0]
    num_right = right.shape[0]
    all = num_left + num_right

    return (num_left / all) * entropy(left) + (num_right / all) * entropy(right)

def gain(all, left, right):
    return entropy(all) - remainder(left, right)

def split_data(dataset, attribute, value):
    # sort by col value
    sorted_dataset = dataset[dataset[:,attribute].argsort()]

    left = sorted_dataset[sorted_dataset[:,attribute] < value]
    right = sorted_dataset[sorted_dataset[:,attribute] > value]

    return (left, right)

def decision_tree_learning(dataset, depth):
    label_col = len(dataset[0]) - 1
    same_class = np.all(dataset[0][label_col] == dataset[:,label_col])

    if same_class:
        node = {"attribute": None, "value": dataset[0][label_col], "left": None, "right": None, "leaf": True}
        return (node, depth)
    else:
        (value, attribute) = find_split(dataset)
        (left, right) = split_data(dataset, attribute, value)
        (left_branch, left_depth) = decision_tree_learning(left, depth + 1)
        (right_branch, right_depth) = decision_tree_learning(right, depth + 1)
        node = {"attribute": attribute, "value": value, "left": left_branch, "right": right_branch}
        return (node, max(left_depth, right_depth))
