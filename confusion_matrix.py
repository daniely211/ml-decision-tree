import numpy as np

def confusion_matrix(labels):
    '''
    :param labels = [(ground_truth1, prediction1), (ground_truth2, prediction2), ...]
    '''
    confusion_matrix = np.zeros((4,4),dtype = int)

    for pair in labels:
        ground_truth = pair[0]
        prediction = pair[1]
        confusion_matrix[ground_truth-1][prediction-1] += 1

    return confusion_matrix

# input: confusion_matrix
# output: an array of true positive per class
def TP(confusion_matrix):
    tp = np.zeros(4, dtype = int)
    for i in range(0,4):
        tp[i] = confusion_matrix[i][i]
    return tp

# output: an array of true negtive per class
def TN(confusion_matrix):
    tn = np.zeros(4, dtype = int)
    sum = np.sum(TP(confusion_matrix))
    for i in range(0,4):
        tn[i] = sum - confusion_matrix[i][i]
    return tn

# output: an array of false positive per class
def FP(confusion_matrix):
    fp = np.zeros(4, dtype = int)
    sum = np.zeros(4, dtype = int)
    for i in range(0,4):
        sum[i] = np.sum(confusion_matrix[:,i])
        fp[i] = sum[i] - confusion_matrix[i][i]
    return fp

# output: an array of false negtive per class
def FN(confusion_matrix):
    fn = np.zeros(4, dtype = int)
    sum = np.zeros(4, dtype = int)
    for i in range(0,4):
        sum[i] = np.sum(confusion_matrix[i,:])
        fn[i] = sum[i] - confusion_matrix[i][i]
    return fn

def recall(cm):
    return sum(TP(cm) / (TP(cm) + FN(cm)))/4

def precision(cm):
    return sum(TP(cm) / (TP(cm) + FP(cm)))/4

def classification_rate(cm):
    return sum((TP(cm) + TN(cm)) / (TP(cm) + TN(cm) + FP(cm) +FN(cm)))/4

def F1_measure(cm):
    return 2 * precision(cm) * recall(cm) / (precision(cm) + recall(cm))

# labels = np.array([(1,1), (1,2), (2,3), (3,3), (3,2), (4,1), (4,4)])
# print(recall(confusion_matrix(labels)))
# print(precision(confusion_matrix(labels)))
# print(classification_rate(confusion_matrix(labels)))
# print(F1_measure(confusion_matrix(labels)))
