import matplotlib.pyplot as plt
from ast import literal_eval
import decision_trees
import numpy as np

decisionNode = dict(boxstyle="round4", fc="w", color='dodgerblue')
leafNode = dict(boxstyle="round4", fc="w", color='dodgerblue')
arrow_args = dict(arrowstyle="-", color='gold', connectionstyle="arc3")

def plot_node(nodeText, centerPt, parent_pt, nodeType):
    plot.ax1.annotate(
        nodeText, xy=parent_pt, xycoords='axes fraction', xytext=centerPt,
        textcoords='axes fraction', va='bottom', ha='center', bbox=nodeType,
        arrowprops=arrow_args
    )

def get_num_leafs(tree):
    num_leafs = 0
    first_list = list(tree.keys())
    first_str = first_list[0]
    second_dict = tree[first_str]

    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1

    return num_leafs


def get_tree_depth(tree):
    max_depth = 0
    first_list = list(tree.keys())
    first_str = first_list[0]
    second_dict=tree[first_str]

    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1

        if this_depth > max_depth:
            max_depth = this_depth

    return max_depth


def plot_tree(tree, parent_pt, nodeTxt):
    num_leafs = get_num_leafs(tree)
    depth = get_tree_depth(tree)
    first_list = list(tree.keys())
    first_str = first_list[0]
    center_point = (plot_tree.xOff + (1.0 + float(num_leafs)) / 2.0 / plot_tree.totalW, plot_tree.yOff)
    plot_node(first_str, center_point, parent_pt, decisionNode)
    second_dict = tree[first_str]
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD

    for key in second_dict.keys():
        if type(second_dict[key]).__name__=='dict':
            plot_tree(second_dict[key], center_point, str(key))
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
            plot_node(second_dict[key], (plot_tree.xOff, plot_tree.yOff), center_point, leafNode)

    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD

def plot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.totalW = float(get_num_leafs(inTree))
    plot_tree.totalD = float(get_tree_depth(inTree))
    plot_tree.xOff = -0.5 / plot_tree.totalW
    plot_tree.yOff = 1.0
    plot_tree(inTree, (0.5, 1.0), '')
    plt.show()

def retrieve_tree(tree):
    if not tree:
        return None

    left = combine(tree.get('left'))
    right = combine(tree.get('right'))
    tree['left'] = left
    tree['right'] = right

    if type(left).__name__ == 'dict':
        retrieve_tree(list(left.values())[0])

    if type(right).__name__ == 'dict':
        retrieve_tree(list(right.values())[0])

    return tree

def combine(tree):
    leaf = tree.get('leaf');
    value = tree.pop('value')
    attribute = tree.pop('attribute')

    new = {}
    if leaf == None:
        condition = 'x' + str(attribute) + '<' + str(value)
        new[condition] = tree
        return new
    else:
        return 'leaf: ' + str(value)

clean_dataset = np.loadtxt('wifi_db/clean_dataset.txt')
(tree, depth) = decision_trees.decision_tree_learning(clean_dataset, 0)
decision_tree = combine(retrieve_tree(tree))

plot(decision_tree)
