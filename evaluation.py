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

    for i in range(k):
        (training_data, test_data) = k_fold_split(dataset, k, i)
        (trained_model, depth) = decision_tree_learning(training_data, 0)

        labels_predictions = [(prediction(trained_model, row), row[room_index]) for row in test_data]
        cm = confusion_matrix(labels_predictions)

        print("Confustion matrix: ")
        print(cm)
        print("Recall: " + str(recall(cm)))
        print("Precision: " + str(precision(cm)))
        print("classification_rate: "+str(classification_rate(cm)))
        print("F1_measure: "+ str(F1_measure(cm)))

        # count = 0
        # for (a, b) in labels_predictions:
        #     if a == b:
        #         count += 1
        #
        # print(count)

        # evaluation goes here
        # c_matrix = confussion_matrix(labels_predictions)

def k_fold_split(dataset, k, index):
    test_size = int(len(dataset) / k)
    start_index = index * test_size
    end_index = (index + 1) * test_size - 1

    data_before = np.array(dataset[0:start_index])
    data_after = np.array(dataset[end_index:len(dataset) - 1])
    training_data = np.concatenate((data_before, data_after), axis=0)

    # training_data = np.ndarray.flatten(np.array([data_before, data_after]))
    test_data = np.array(dataset[start_index:end_index + 1])

    return (training_data, test_data)

# def confussion_matrix(labels_predictions):
#   pass

evaluation(clean_dataset)
