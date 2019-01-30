from numpy.random import shuffle
from decision_tree import decision_tree_learning
import numpy as np
from confusion_matrix import confusion_matrix, recall, precision, classification_rate, F1_measure

clean_dataset = np.loadtxt('wifi_db/clean_dataset.txt')
noisy_dataset = np.loadtxt('wifi_db/noisy_dataset.txt')
room_index = 7

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

    for i in range(k-1): #k-1 for validation
        # (training_data, test_data) = k_fold_split(dataset, k, i)
        # Split the data into training, test, validation
        (training_data, test_data, validation_data) = k_fold_validation_split(dataset, k, i)
        # Build the model with the training data
        (trained_model, depth) = decision_tree_learning(training_data, 0)

        labels_predictions = [(prediction(trained_model, row), row[room_index]) for row in validation_data]

        cm_before_pruning = confusion_matrix(labels_predictions)
        recall_before_pruning = recall(cm_before_pruning)
        precision__before_pruning = precision(cm_before_pruning)
        cr_before_pruning = classification_rate(cm_before_pruning)
        F1_before_pruning = F1_measure(cm_before_pruning)

        #BEFORE PRUNING
        print("Before Pruning Confustion matrix: ")
        print(cm)
        print("Recall: " + str(recall_before_pruning))
        print("Precision: " + str(precision__before_pruning))
        print("classification_rate: "+str(cr_before_pruning))
        print("F1_measure: "+ str(F1_before_pruning)+"\n\n")

        # Prune the model until all the metrics increase
        # TODO IMPLEMENT PRUNING
        # TODO Print out the metrics again and also the difference

        #DFS 
        # While there are node with 2 leaves not visited
        #   prune the node
        #   run validation test data on the new model
        #   Check if it has imporved
        #       Keep the pruning if it has
        #       Revert the prune if it has not

        


        # Prune the trained trained_model
        # Common approach with decision trees
        # Go through all the nodes that are only connected to leaves and check if the accuracy on the validation dataset would increase if this node is turned into a leaf.
        # You need to do this recursively as when you turn nodes into leaves, you might create new nodes that are connected to two leaves.

        #AFTER PRUNING

def dfs(root):
    visited = []
    stack = [root]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            # new nodes are added to the start of stack
            stack = graph[vertex] - visited + stack 
    return visited


def k_fold_split(dataset, k, index):
    test_size = int(len(dataset) / k)
    start_index = index * test_size
    end_index = (index + 1) * test_size - 1

    data_before = np.array(dataset[0:start_index])
    data_after = np.array(dataset[end_index:len(dataset) - 1])
    training_data = np.concatenate((data_before, data_after), axis=0)

    test_data = np.array(dataset[start_index:end_index + 1])

    return (training_data, test_data)

def k_fold_validation_split(dataset, k, index):
    split_size = int(len(dataset) / k)
    validation_start_index = (index + 1) * split_size
    validation_end_index = validation_start_index + split_size

    data_before = np.array(dataset[split_size:validation_start_index])
    data_after = np.array(dataset[validation_end_index:len(dataset)-1])
    training_data = np.concatenate((data_before, data_after), axis=0)

    test_data = np.array(dataset[0:split_size])

    validation_data = np.array(dataset[validation_start_index: validation_end_index])

    return (training_data, test_data, validation_data)


# def confussion_matrix(labels_predictions):
#   pass

evaluation(clean_dataset)
# evaluation(noisy_dataset)


if __name__ == "__main__":


    pass
