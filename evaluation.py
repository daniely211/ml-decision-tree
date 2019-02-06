from numpy.random import shuffle
from decision_tree import decision_tree_learning
import numpy as np
from confusion_matrix import confusion_matrix, recall, precision, classification_rate, F1_measure

clean_dataset = np.loadtxt('wifi_db/clean_dataset.txt')
noisy_dataset = np.loadtxt('wifi_db/noisy_dataset.txt')
room_index = 7


def get_cr(model, data):
    labels_predictions = [(prediction(model, row), row[room_index]) for row in data]
    cm = confusion_matrix(labels_predictions)
    return classification_rate(cm)

def prediction(node, row):
    
    if node['leaf']:
        return node['value']

    if row[node['attribute']] < node['value']:
        return prediction(node['left'], row)
    else:
        return prediction(node['right'], row)

def evaluation(dataset):

    shuffle(dataset)
    k = 10
    j = 9
    avg_difs = np.array([])
    avg_pruned = np.array([])
    avg_unpruned = np.array([])
    for test_i in range(k):
        # Split the data into training + validation, test
        (training_validation_data, test_data) = k_fold_split(dataset, k, test_i)
    
        pruned_test_scores = np.array([]) # this is the average test score for all the pruned trees combined
        unpruned_test_scores = np.array([])

        for validation_i in range(k):
            # Split the data into training, validation
            (training_data, validation_data) = k_fold_split(training_validation_data, k, validation_i)
            # Build the model with the training data

            (trained_model, _) = decision_tree_learning(training_data, 0)

            test_score_before = get_cr(trained_model, test_data)
            unpruned_test_scores = np.append(unpruned_test_scores, test_score_before)
            # print ("Test score for the UNpruned tree: "+ str(test_score_before))
            cr_before_pruning = get_cr(trained_model, validation_data)

            # print("Before Pruning: ")

            (pruned_tree, depth) = prune_tree(trained_model, validation_data)
            test_score = get_cr(pruned_tree, test_data)

            # we get the validation error for the final pruned tree
            # validation_error = 1 - test_score
            pruned_test_scores = np.append(pruned_test_scores, test_score)
            # print("final test score for this prune tree: " + str(test_score))
            # print("the difference between the scores :" + str(test_score - test_score_before))

        pruned_avg = np.mean(pruned_test_scores)
        unpruned_avg = np.mean(unpruned_test_scores)
        avg_dif = pruned_avg - unpruned_avg
        avg_pruned = np.append(avg_pruned, pruned_avg)
        avg_unpruned = np.append(avg_unpruned, unpruned_avg)
        avg_difs = np.append(avg_difs, avg_dif)
        print("********* RESULT **********")
        print("Average difference between pruned and unpruned ="+str(avg_dif) + "\n\n")

    print("************************ FINAL RESULT **********************")
    print("The Average unpruned Classification rate =" + str(np.mean(avg_unpruned)))
    print("The Average pruned Classification rate ="+str(np.mean(avg_pruned)))
    print("The Average differences across all 10 test folds ="+str(np.mean(avg_difs)))
    print("Happy Chinese New Year!")

def prune_tree(node, validation_data, parent=None, parent_side=None, root=None, depth=0):
    if root is None:
        root = node

    if node['leaf']:
        return (node, depth)

    # Our pruning algorithm does a depth first search and prune the left most node first then check the rest.
    (node['left'], depth_left) = prune_tree(node['left'], validation_data, node, 'left', root, depth + 1)
    (node['right'], depth_right) = prune_tree(node['right'], validation_data, node, 'right', root, depth + 1)

    if parent and node['left'] and node['left']['leaf'] and node['right'] and node['right']['leaf']:
        # Replace the courrent node to be the majority and then see if that improves the CR
        cr_before_pruning = get_cr(root, validation_data)

        # replace the node to be the majority and see if the CR imporves
        pruned_node = {"attribute": None, "value": node['mode_class'], "left": None, "right": None, "leaf": True}
        parent[parent_side] = pruned_node

        # See if the prunning has increased the cr_before pruning
        cr_after_pruning = get_cr(root, validation_data)
        parent[parent_side] = node  # reset the parent node to the original

        if cr_after_pruning <= cr_before_pruning:
            # Prunning did not improve the CR
            return node, max(depth_left, depth_right)
        else:
            # print("Before pruning:" + str(cr_before_pruning))
            # print("After pruning:" + str(cr_after_pruning))
            # Pruning improved the score so we return the leaf with the majority.
            return pruned_node, max(depth_left, depth_right)

    # not a node with 2 leafs simply return here
    return node, max(depth_left, depth_right)

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
# evaluation(noisy_dataset)

