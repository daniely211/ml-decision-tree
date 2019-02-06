from numpy.random import shuffle
from decision_tree import decision_tree_learning
import numpy as np
from confusion_matrix import confusion_matrix, recall, precision, classification_rate, F1_measure

clean_dataset = np.loadtxt('wifi_db/clean_dataset.txt')
noisy_dataset = np.loadtxt('wifi_db/noisy_dataset.txt')
room_index = 7

def get_cr(tree, data):
    labels_predictions = [(prediction(tree, row), row[room_index]) for row in data]
    cm = confusion_matrix(labels_predictions)

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
            (trained_tree, _) = decision_tree_learning(training_data)

            # print(trained_tree)
            # print("**********************************************")

            test_score_before = get_cr(trained_tree, test_data)
            unpruned_test_scores = np.append(unpruned_test_scores, test_score_before)

            (pruned_tree, depth) = prune_tree(trained_tree, validation_data)

            # print(pruned_tree)
            print("**********************************************")

            test_score_after = get_cr(pruned_tree, test_data)

            pruned_test_scores = np.append(pruned_test_scores, test_score_after)

        pruned_avg = np.mean(pruned_test_scores)
        unpruned_avg = np.mean(unpruned_test_scores)
        avg_dif = pruned_avg - unpruned_avg
        avg_pruned = np.append(avg_pruned, pruned_avg)
        avg_unpruned = np.append(avg_unpruned, unpruned_avg)
        avg_difs = np.append(avg_difs, avg_dif)
        print("********* RESULT **********")
        print("Average difference between pruned and unpruned: " + str(avg_dif) + "\n\n")

    print("************************ FINAL RESULT **********************")
    print("average unpruned classification rate: " + str(np.mean(avg_unpruned)))
    print("average pruned classification rate: " + str(np.mean(avg_pruned)))
    print("average differences across all 10 test folds: " + str(np.mean(avg_difs)))

def prune_tree(node, validation_data, parent=None, parent_side=None, root=None, depth=0):
    if root is None:
        root = node

    if node['leaf']:
        return (node, depth)

    # Our pruning algorithm does a depth first search and prune the left most node first then check the rest.
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
            return (new_leaf, max(depth_left, depth_right))

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
# (dt, depth) = decision_tree_learning(clean_dataset)
# print(isEqual(dt, dt))
