from numpy.random import shuffle
from decision_tree import decision_tree_learning
import numpy as np
from confusion_matrix import confusion_matrix, recall, precision, classification_rate, F1_measure

clean_dataset = np.loadtxt('wifi_db/clean_dataset.txt')
noisy_dataset = np.loadtxt('wifi_db/noisy_dataset.txt')
room_index = 7

def get_confusion_matrix(tree, data):
    labels_predictions = [(prediction(tree, row), row[room_index]) for row in data]
    return confusion_matrix(labels_predictions)

def get_cr(tree, data):
    cm = get_confusion_matrix(tree, data)
    cr = 0

    for i in range(4):
        cr += classification_rate(cm, i)

    return cr / 4

def prediction(node, row):
    if node['leaf']:
        return node['value']

    if row[node['attribute']] < node['value']:
        return prediction(node['left'], row)
    else:
        return prediction(node['right'], row)

def evaluation(dataset):
    print("Shuffling dataset...")
    shuffle(dataset)

    k = 10
    avg_difs = np.array([])
    avg_pruned = np.array([])
    avg_unpruned = np.array([])

    cm = np.zeros((4, 4), dtype=int)
    avg_cr = np.array([])
    avg_precision = np.array([])
    avg_recall = np.array([])
    avg_f1 = np.array([])

    for test_i in range(k):
        print("Splitting into training & validation fold and test fold...")
        # split the data into training + validation, test
        (training_validation_data, test_data) = k_fold_split(dataset, k, test_i)

        pruned_test_scores = np.array([])
        unpruned_test_scores = np.array([])

        for validation_i in range(k - 1):
            # split the data into training, validation
            (training_data, validation_data) = k_fold_split(training_validation_data, k - 1, validation_i)

            # build the model with the training data
            (trained_tree, depth_before) = decision_tree_learning(training_data)
            test_score_before = get_cr(trained_tree, test_data)
            unpruned_test_scores = np.append(unpruned_test_scores, test_score_before)

            # prune the tree
            (pruned_tree, depth_after) = prune_tree(trained_tree, validation_data)

            # get classification rate on pruned tree
            test_score_after = get_cr(pruned_tree, test_data)
            pruned_test_scores = np.append(pruned_test_scores, test_score_after)

            # calculate metrics
            new_cm = get_confusion_matrix(pruned_tree, test_data)
            cm = np.add(cm, new_cm)

            

            avg_cr = np.append(avg_cr, test_score_after)
            avg_precision = np.append(avg_precision, test_score_after)
            avg_recall = np.append(avg_recall, recall(new_cm))

        unpruned_avg = np.mean(unpruned_test_scores)
        pruned_avg = np.mean(pruned_test_scores)
        diff = np.subtract(pruned_avg, unpruned_avg)
        avg_dif = np.mean(diff)
        avg_pruned = np.append(avg_pruned, pruned_avg)
        avg_unpruned = np.append(avg_unpruned, unpruned_avg)
        avg_difs = np.append(avg_difs, avg_dif)

        print("********* RESULT **********")
        print("average unpruned test data cr: " + str(np.mean(unpruned_test_scores)))
        print("average pruned test data cr: " + str(np.mean(pruned_test_scores)))
        print("Average difference between pruned and unpruned: " + str(avg_dif))

    average_cm = np.true_divide(cm, k * (k - 1))

    print("************************ FINAL RESULT **********************")
    print("average unpruned classification rate: " + str(np.mean(avg_unpruned)))
    print("average pruned classification rate: " + str(np.mean(avg_pruned)))
    print("average differences across all 10 test folds: " + str(np.mean(avg_difs)))
    print("average classification rate: " + str(np.mean(avg_cr)))
    print("average precision: " + str(np.mean(avg_precision)))
    print("average recall: " + str(np.mean(avg_recall)))
    print("average f1: " + str(np.mean(avg_f1)))
    print("average confusion matrix:")
    print(average_cm)


def prune_tree(node, validation_data, parent=None, parent_side=None, root=None, depth=0):
    if root is None:
        root = node

    if node['leaf']:
        return (node, depth)

    # perform a depth first search to allow nodes and their parents to be
    # recursively pruned
    (node['left'], depth_left) = prune_tree(node['left'], validation_data, node, 'left', root, depth + 1)
    (node['right'], depth_right) = prune_tree(node['right'], validation_data, node, 'right', root, depth + 1)

    # conditions to allow a node to be pruned
    if parent and node['left'] and node['left']['leaf'] and node['right'] and node['right']['leaf']:

        # capture classification rate before pruning
        cr_before_pruning = get_cr(root, validation_data)

        # replace the current node with a leaf who's value is the most common
        # classification
        new_leaf = {
            "attribute": None,
            "value": node['mode_class'],
            "left": None,
            "right": None,
            "leaf": True
        }
        parent[parent_side] = new_leaf

        # capture classification rate after pruning
        cr_after_pruning = get_cr(root, validation_data)

        # check if pruning the node improves the classification rate
        if cr_after_pruning < cr_before_pruning:
            # classification error worsened, reset to original node
            return (node, max(depth_left, depth_right))
        else:
            # classification rate improved, replace pruned node with newly
            # created leaf
            return (new_leaf, max(depth_left, depth_right) - 1)

    # not a node with 2 leafs simply return here
    return (node, max(depth_left, depth_right))

def k_fold_split(dataset, k, index):
    test_size = int(len(dataset) / k)
    start_index = index * test_size
    end_index = (index + 1) * test_size - 1

    data_before = np.array(dataset[0:start_index])
    data_after = np.array(dataset[end_index:len(dataset) - 1])
    training_data = np.concatenate((data_before, data_after), axis=0)

    test_data = np.array(dataset[start_index:end_index + 1])

    return (training_data, test_data)

# evaluation(clean_dataset)
evaluation(noisy_dataset)
