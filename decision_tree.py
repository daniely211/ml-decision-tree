import numpy as np

np.set_printoptions(threshold=float('inf'))

clean_dataset = np.loadtxt('wifi_db/clean_dataset.txt')
noisy_dataset = np.loadtxt('wifi_db/noisy_dataset.txt')

# Find the best split to perform
#
#   dataset     the dataset to find the best split point on
def find_split(dataset):
    max_gain = float('-inf')
    split_point_attr = None
    class_index = len(dataset[0]) - 1

    dataset.astype(float)

    # iterate over each attribute and find a split point with the largest gain
    for i in range(len(dataset[0]) - 1):
        # sort whole dataset by current attribute (i)
        sorted_dataset = dataset[dataset[:,i].argsort()]

        # find all possible split points
        split_points = find_split_points(sorted_dataset, i)

        # return the split point with the best gain
        (gain, split_point) = find_best_gain(split_points, sorted_dataset, i)

        if gain > max_gain:
            max_gain = gain
            split_point_attr = (split_point, i)

    return split_point_attr

# Find all possible split points for the given attribute
#
#   sorted_dataset      the dataset, sorted on the given attribute in ascending
#                       order
#   attribute           the attribute to find the split points on
def find_split_points(sorted_dataset, attribute):
    split_points = np.array([])

    class_index = len(sorted_dataset[0]) - 1

    for i in range(len(sorted_dataset) - 1):
        val = sorted_dataset[i][attribute]
        classification = sorted_dataset[i][class_index]
        val_next = sorted_dataset[i + 1][attribute]
        classification_next = sorted_dataset[i + 1][class_index]

        if val != val_next and classification != classification_next:
            mid = (val + val_next) / 2
            split_points = np.append(split_points, mid)

    return split_points

# Finds the largest gain given some split points and a sorted column
#
#   split_points    array of all possible split points
#   sorted_dataset  the dataset, sorted on the given attribute in ascending
#                   order
#   attribute       the attribute which will be split
def find_best_gain(split_points, sorted_dataset, attribute):
    max_gain = float('-inf')
    gain_point = (float('-inf'), None)

    for split_point in split_points:
        (left, right) = split_data(sorted_dataset, attribute, split_point)
        curr_gain = gain(sorted_dataset, left, right)

        if curr_gain > max_gain:
            max_gain = curr_gain
            gain_point = (curr_gain, split_point)

    return gain_point

# Calculate entropy of the dataset
#
#   dataset     the dataset to calculate entropy for
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

# Calculate the entropy of the left/right split
#
#   left    the left split
#   right   the right split
def remainder(left, right):
    num_left = left.shape[0]
    num_right = right.shape[0]
    all = num_left + num_right

    return (num_left / all) * entropy(left) + (num_right / all) * entropy(right)

# Calculates the information gain of a split
#
#   all     full dataset before split
#   left    left split
#   rigth   right split
def gain(all, left, right):
    return entropy(all) - remainder(left, right)

# Split the given dataset by the value of a certain attribute, anything below
# the value goes to the left fold, anything above goes to the right fold.
#
# NOTE: this function assumes 'dataset' is sorted in ascending order on the
# given 'attribute'
#
#   dataset     the dataset to be split
#   attribute   the attribute to split on
#   value       the value to split on
def split_data(dataset, attribute, value):
    left = dataset[dataset[:,attribute] < value]
    right = dataset[dataset[:,attribute] > value]

    return (left, right)

# Return the most common classification for a given dataset
#
#   dataset     the dataset of classificatons
#   label_col   the column which contains the classifications
def find_most_common_class(dataset, label_col):
    bin_count = np.bincount(dataset[:,label_col].astype(int))
    return float(np.argmax(bin_count))

# Recursively builds a decision tree from a given dataset
#
#   dataset     the dataset to use to build the tree
#   depth       the depth of the tree, defaults to 0, will be incremented on
#               recursive calls
def decision_tree_learning(dataset, depth=0):
    label_col = len(dataset[0]) - 1
    same_class = np.all(dataset[0][label_col] == dataset[:,label_col])

    if same_class:
        # all data in dataset has the same classification, create a leaf node
        # with this classification
        node = {
            "attribute": None,
            "value": dataset[0][label_col],
            "left": None,
            "right": None,
            "leaf": True
        }
        return (node, depth)
    else:
        # every node will store the most popular classification in its children
        # to be used in pruning

        result = find_split(dataset)

        if result == None:
            # could not find a split, collapse node into the most common class
            mode_class = find_most_common_class(dataset, label_col)

            node = {
                "attribute": None,
                "value": mode_class,
                "left": None,
                "right": None,
                "leaf": True
            }
            return (node, depth)
        else:
            # a split is possible
            (value, attribute) = result

            # sort by col value
            sorted_dataset = dataset[dataset[:, attribute].argsort()]

            (left, right) = split_data(sorted_dataset, attribute, value)

            # recursively call on left and right splits
            (left_branch, left_depth) = decision_tree_learning(left, depth + 1)
            (right_branch, right_depth) = decision_tree_learning(right, depth + 1)

            mode_class = find_most_common_class(dataset, label_col)

            node = {
                "attribute": attribute,
                "value": value,
                "left": left_branch,
                "right": right_branch,
                "leaf": False,
                "mode_class": mode_class
            }

            return (node, max(left_depth, right_depth))
