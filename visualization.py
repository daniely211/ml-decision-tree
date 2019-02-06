import matplotlib.pyplot as plt
import decision_tree
import numpy as np
# from evaluation import prune_tree
# from evaluation import k_fold_split
decision_node = dict(boxstyle="round4",fc="w", color ='dodgerblue')
leaf_node = dict(boxstyle="round4",fc="w", color='dodgerblue')
arrow_args = dict(arrowstyle="-",color='gold', connectionstyle="arc3")

def plot_node(nodeText,centerPt,parentPt,nodeType):
    create_plot.ax1.annotate(nodeText, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction',
                           va='bottom', ha='center', bbox=nodeType, arrowprops=arrow_args)


def get_num_leaves(tree):
    '''
    Get the number of leaf nodes
    As a node, if it contains a dictionary set, this node is a leaf node
    So we need to search its left child and right child until we find a leaf node
    '''
    num_leaf_nodes = 0
    key_list = list(tree.keys())
    condition = key_list[0]       # split condition
    content = tree[condition]     # content of the node
    keys = ['left', 'right']

    for key in keys:
        if type(content[key]).__name__ == 'dict':
            num_leaf_nodes += get_num_leaves(content[key])
        else: num_leaf_nodes += 1

    return num_leaf_nodes

def plot_tree(tree, parentPt, nodeTxt, depth):
    '''
    Ploting the tree depends on the depth of the tree
    and the number of leaf nodes
    '''
    num_leaf_nodes = get_num_leaves(tree)
    condition = list(tree.keys())
    content = condition[0]
    # Plot the tree in the center
    centrePt = (plot_tree.xOff + (1.0 + float(num_leaf_nodes))/2.0/plot_tree.totalW, plot_tree.yOff)
    plot_node(content, centrePt, parentPt, decision_node)
    child_node = tree[content]
    plot_tree.yOff = plot_tree.yOff - 1.0/plot_tree.totalD

    keys = ['left', 'right']

    for key in keys:
        #TODO CHANGE THIS
        if type(child_node[key]).__name__=='bool':
            continue

        if type(child_node[key]).__name__=='dict':
            # if it is not a leaf node
            plot_tree(child_node[key], centrePt, str(key), depth)
        else:
            # a leaf node
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
            plot_node(child_node[key], (plot_tree.xOff, plot_tree.yOff), centrePt, leaf_node)

    plot_tree.yOff = plot_tree.yOff + 1.0/plot_tree.totalD


def create_plot(tree, depth):
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    create_plot.ax1 = plt.subplot(111,frameon=False,**axprops)
    plot_tree.totalW = float(get_num_leaves(tree))
    plot_tree.totalD = float(depth)
    plot_tree.xOff = -0.5/plot_tree.totalW
    plot_tree.yOff = 1.0
    plot_tree(tree,(0.5,1.0),'', depth)
    plt.show()

def retrieve_tree(tree):
    if not tree:
        return None

    left = merge_keys(tree.get('left'))
    right = merge_keys(tree.get('right'))
    tree['left'] = left;
    tree['right'] = right;
    if type(left).__name__ == 'dict':
        retrieve_tree(list(left.values())[0])
    if type(right).__name__ == 'dict':
        retrieve_tree(list(right.values())[0])
    return tree

def merge_keys(tree):
    '''
    Merge the keys 'attribute' and 'value' into a new key 'condition'
    For example: {’x3 < -55’: {’left’:{ left_child }, ’right’:{ right_child }}}
    '''
    new = {}

    if tree['leaf']:
        return 'leaf:' + str(tree['value'])
    else:
        condition = 'x' + str(tree['attribute']) + ' < ' + str(tree['value'])
        new[condition] = tree
        return new

clean_dataset = np.loadtxt('wifi_db/clean_dataset.txt')
noisy_dataset = np.loadtxt('wifi_db/noisy_dataset.txt')

# (dt, depth) = decision_tree.decision_tree_learning(clean_dataset, 0)
# decisionTree = merge_keys(retrieve_tree(dt))
# print(decisionTree)
# (training_data, test_data) = k_fold_split(clean_dataset, 10, 3)
# (training_data, test_data) = k_fold_split(noisy_dataset, 10, 3)
# (dt_unpruned, depth_1) = decision_tree.decision_tree_learning(training_data, 0)
# (pruned_tree, depth_2) = prune_tree(dt_unpruned, test_data)

# dt_unpruned_plot = merge_keys(retrieve_tree(dt_unpruned))

# pruned_tree_plot = merge_keys(retrieve_tree(pruned_tree))

# create_plot(dt_unpruned_plot, depth_1)
# create_plot(pruned_tree_plot, depth_2)
