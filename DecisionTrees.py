import numpy as np
clean_dataset = np.loadtxt('wifi_db/clean_dataset.txt')
noisy_dataset = np.loadtxt('wifi_db/noisy_dataset.txt')

def find_split(dataset):
    left = np.array()
    right = np.array()

    gains = np.array()

    # find gain for each attribute
    for i in range(len(dataset[0])):
        values = [(row[i], row[len(row) - 1]) for row in dataset]
        values.sort(key = lambda x: x[0])
        split_points = find_split_points(values)
        gain = find_best_gain(split_points, values)
        gains.append(gain)

    # get attribute with largest gain

    # for EACH router router_strength (attribute)
        # sort the array, find the mid point and midpoint +1 then find average between those 2 numbers
        # split the router streght left = < midpoint, right = < midpoint
        # calculate the GAIN of the split value
        # compare with the largest gain value if it is largest, replace the left and right

    return {"attribute": router_number, "value": value, "left": left, "right": right}

def find_split_points(sorted_col):
    split_points = np.array()

    for i in range(len(sorted_col) - 1):
        (val, classification) = sorted_col[i]
        (val_next, classification_next) = sorted_col[i + 1]

        if val != val_next and classification != classification_next:
            split_points.append((val_next - val) / 2)

    return split_points

def find_best_gain(split_points, sorted_col):
    gains = np.array()

    for split_point in split_points:
        left = list(filter(lambda (x, _): x < split_point, sorted_col))
        right = list(filter(lambda (x, _): x > split_point, sorted_col))
        all = [i for (i, _) in sorted_col]
        gains.append(gain(all, left, right))

    return max(gains)

# Calculate entropy of the dataset
def entropy(dataset):
    allLabel = np.array(dataset[:,-1])
    total_num = dataset.shape[0]
    probLabel = np.zeros(4)
    H = 0

    for i in range(0,4):
        probLabel[i] = np.count_nonzero(allLabel == i+1)/total_num
        # print(probLabel[i])
        H = H - probLabel[i] * np.log2(probLabel[i])

    return H

#print(entropy(clean_dataset))

def remainder(sLeft, sRight):
    numLeft = np.shape(sLeft)[0]
    numRight = np.shape(sRight)[0]

    r = numLeft / (numLeft + numRight) * entropy(sLeft) + numRight / (numLeft + numRight) * entropy(sRight)# remainder

    return r

def gain(all, left, right):
    return entropy(all) - remainder(left, right)


def visualizeTree(dataset):


def decision_tree_learning(dataset, depth_variable):
    # if all samples have same label
