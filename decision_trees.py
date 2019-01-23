import numpy as np

clean_dataset = np.loadtxt('wifi_db/clean_dataset.txt')
noisy_dataset = np.loadtxt('wifi_db/noisy_dataset.txt')

# Find the best split to perform
def find_split(dataset):
    left = np.array([])
    right = np.array([])
    gains = np.array([])

    class_index = len(dataset[0]) - 1

    # find gain for each attribute
    for i in range(len(dataset[0]) - 1):
        # create array of [col value, label]
        labelled_col = dataset[:, [i, class_index]]

        # sort by col value
        labelled_col = labelled_col[labelled_col[:,0].argsort()]

        split_points = find_split_points(labelled_col)
        gain = find_best_gain(split_points, labelled_col)
        gains = np.append(gains, gain)

    # get attribute with largest gain

# Find all possible split points for the given column
def find_split_points(sorted_col):
    split_points = np.array([])

    for i in range(len(sorted_col) - 1):
        (val, classification) = tuple(sorted_col[i])
        (val_next, classification_next) = tuple(sorted_col[i + 1])

        if val != val_next and classification != classification_next:
            diff = (val_next - val) / 2
            mid = val_next + diff
            split_points = np.append(split_points, mid)

    return split_points

# Finds the largest gain given some split points and a sorted column
def find_best_gain(split_points, sorted_col):
    gains = np.array([])

    for split_point in split_points:
        left = sorted_col[sorted_col[:,0] < split_point]
        right = sorted_col[sorted_col[:,0] > split_point]
        gains = np.append(gains, gain(sorted_col, left, right))

    return max(gains)

# Calculate entropy of the dataset
def entropy(dataset):
    classes = dataset[:,1]

    H = 0

    for i in range(1, 4):
        pk = (classes == i).sum() / len(classes)
        if pk != 0:
            H = H - pk * np.log2(pk)

    return H

def remainder(left, right):
    num_left = left.shape[0]
    num_right = right.shape[0]
    all = num_left + num_right

    return (num_left / all) * entropy(left) + (num_right / all) * entropy(right)

def gain(all, left, right):
    return entropy(all) - remainder(left, right)


# def visualizeTree(dataset):


# def decision_tree_learning(dataset, depth_variable):
    # if all samples have same label

print(find_split(clean_dataset))
