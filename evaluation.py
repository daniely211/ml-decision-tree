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

#check if the trees are identical
def isEqual(tree1, tree2):
    if tree1 == tree2:
        return True # TIM FIX THIS
    attri = (tree1['attribute'] == tree2['attribute'])
    print(attri)
    val = (tree1['value'] == tree2['value'])
    print(val)
    leaf = (tree1['leaf'] == tree2['leaf'])
    print (leaf)
    cnt = (tree1['count'] == tree2['count'])
    print(cnt)
    return attri and val and leaf and cnt and isEqual(tree1['left'], tree2['left']) and isEqual(tree1['right'], tree2['right'])

def evaluation(dataset):
    shuffle(dataset)
    k = 10
    j = 9

    for test_i in range(k): #k-1 for validation
        # Split the data into training + validation, test
        (training_validation_data, test_data) = k_fold_split(dataset, k, test_i)
    
        test_scores = [] # this is the average test score for all the pruned trees combined

        for validation_i in range(k):
            # Split the data into training, validation
            (training_data, validation_data) = k_fold_split(training_validation_data, k, validation_i)
            # Build the model with the training data
            (trained_model, depth) = decision_tree_learning(training_data, 0)

            # labels_predictions_before_pruning = get_prediction(trained_model, validation_data)
            # # Here we calculate te CR before the pruning, and pass it into the prune function to compare with the pruned tree
            # cm_before_pruning = confusion_matrix(labels_predictions_before_pruning)
            cr_before_pruning = get_cr(trained_model, validation_data)

            print("Before Pruning: ")
            print("classification_rate: " + str(cr_before_pruning))
            print("\n\n")

            pruned_tree = prune_tree(trained_model, cr_before_pruning, validation_data)

            # prune it the second time to see if there are any changes

            new_tree = prune_tree(pruned_tree, get_cr(pruned_tree, validation_data), validation_data)

            # we will now start to prune this trained model until it does not change

            while not isEqual(new_tree, pruned_tree):
                pruned_tree = new_tree  # swap the previous one
                cr_new_tree = get_cr(pruned_tree, validation_data)
                new_tree = prune_tree(new_tree, cr_new_tree, validation_data)
            
            # The new_tree is the Final pruned model.
            # we obtain the test_score of the final pruned model for this validation data set

            test_score = get_cr(new_tree, test_data)

            # we get the validation error for the final pruned tree

            validation_error = 1 - test_score
            test_scores.append(test_score)
            print("final test score for this prune tree: " + str(test_score))

    

def prune_tree(node, cr_before_pruning, validation_data, parent=None, parent_side=None, root=None):
    if root is None:
        root = node

    if node['leaf']:
        return node

    # Our pruning algorithm does a depth first search and prune the left most node first then check the rest.
    node['left'] = prune_tree(node['left'], cr_before_pruning, validation_data, node, 'left', root)
    node['right'] = prune_tree(node['right'], cr_before_pruning, validation_data, node, 'right', root)

    if parent and node['left'] and node['left']['leaf'] and node['right'] and node['right']['leaf']:
        #FIND THE MAJORITY OF LEFT AND RIGHT
        if node['left']['count'] > node['right']['count']:
            parent[parent_side] = node['left'] # This alters the root in place
        else:
            parent[parent_side] = node['right'] # This alters the root in place
        
        # See if the prunning has increased the cr_before pruning
        cr_after_pruning = get_cr(root, validation_data)

        if cr_after_pruning <= cr_before_pruning:
            # Prunning did not improve the CR
            parent[parent_side] = node # reset the parent node to the original.

    return node

test_tree = {
    'left': {
        'left': { 'leaf': True },
        'right': {
            'left': { 'leaf': True, 'value': 1 },
            'right': { 'leaf': True, 'value': 2 },
            'leaf': False
        },
        'leaf': False
    },
    'right': {
        'left': { 'leaf': True },
        'right': { 'leaf': True },
        'leaf': False
    },
    'leaf': False
}
# print(test_tree)
# print(prune_tree(test_tree))

def k_fold_split(dataset, k, index):
    test_size = int(len(dataset) / k)
    start_index = index * test_size
    end_index = (index + 1) * test_size - 1

    data_before = np.array(dataset[0:start_index])
    data_after = np.array(dataset[end_index:len(dataset) - 1])
    training_data = np.concatenate((data_before, data_after), axis=0)

    test_data = np.array(dataset[start_index:end_index + 1])

    return (training_data, test_data)

# def confussion_matrix(labels_predictions):
#   pass

# evaluation(clean_dataset)
evaluation(noisy_dataset)
# (dt, depth) = decision_tree_learning(clean_dataset, 0)
# print(isEqual(dt, dt))

