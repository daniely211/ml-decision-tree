import matplotlib.pyplot as plt
import decision_tree
import numpy as np
from evaluation import prune_tree, k_fold_split
from numpy.random import shuffle


class PlotTree:

    def __init__(self, tree, depth):
        self.axes = plt.subplot(111, frameon=False, **dict(xticks=[],yticks=[]))
        self.depth = float(depth)
        self.width = self.get_width(tree)
        self.x_loc = -1.0/self.width
        self.y_loc = 1.0


    def plot_node(self, text, loc, parent, node_type):
        '''
        define the type and color of annotations and node
        '''
        args = dict(arrowstyle = "-", color = 'gold', connectionstyle = "arc3")
        self.axes.annotate(text, xy = parent, xycoords = 'axes fraction', xytext = loc, textcoords = 'axes fraction',
                           va = 'bottom', ha = 'center', bbox = node_type, arrowprops = args)


    def get_width(self, tree):
        '''
        Get the number of leaf nodes
        As a node, if it contains a dictionary set, this node is a leaf node
        So we need to search its left child and right child until we find a leaf node
        '''
        num_leaf_nodes = 0

        root_decision = list(tree.keys())[0]
        node = tree[root_decision]

        for key in ['left', 'right']:
            try:
                num_leaf_nodes += self.get_width(node[key])
            except:
                num_leaf_nodes += 1

        return num_leaf_nodes


    def plot(self, tree, parent, text):
        '''
        Ploting the tree depends on the depth of the tree
        and the number of leaf nodes
        '''
        num_leaf_nodes = self.get_width(tree)

        root_decision = list(tree.keys())[0]
        node = tree[root_decision]

        # Plot the tree in the center
        x_replacement = (1.0 + float(num_leaf_nodes)) / 2.0 / self.width
        centrePt = (self.x_loc + x_replacement , self.y_loc)
        decision_node = dict(boxstyle = "round4", fc = "w", color = 'dodgerblue')
        self.plot_node(root_decision, centrePt, parent, decision_node)

        self.y_loc -= 1.0/self.depth

        for key in ['left', 'right']:
            try:
                # if it is not a leaf node
                self.plot(node[key], centrePt, str(key))
            except:
                # a leaf node
                self.x_loc += 1.0 / self.width
                leaf_node = dict(boxstyle = "round4", fc = "w", color = 'green')
                self.plot_node(node[key], (self.x_loc, self.y_loc), centrePt, leaf_node)

        self.y_loc += 1.0/self.depth


def retrieve_tree(tree):
    '''
    Get the left child and right child of the decision node
    '''
    if not tree:
        return None

    for key in ['left', 'right']:
        child = merge_keys(tree[key])
        tree[key] = child
        if type(child).__name__ == 'dict':
            retrieve_tree(list(child.values())[0])

    return tree


def merge_keys(tree):
    '''
    Merge the keys 'attribute' and 'value' into a new key 'condition'
    For example:
    if a parent node {’x3 < -55’: {'attribute': ..., 'value': ..., ’left’:{ left_child }, ’right’:{ right_child }}}
    else a leaf node 'leaf: ...'
    '''
    new = {}

    if tree['leaf']:
        return 'leaf:' + str(tree['value'])
    else:
        condition = 'x' + str(tree['attribute']) + ' < ' + str(tree['value'])
        new[condition] = tree
        return new


def main():
    # upload clean and noisy dataset
    clean_dataset = np.loadtxt('wifi_db/clean_dataset.txt')
    noisy_dataset = np.loadtxt('wifi_db/noisy_dataset.txt')
    shuffle(clean_dataset)
    shuffle(noisy_dataset)


    # plot initial tree with clean dataset
    (dt_clean, depth_1o) = decision_tree.decision_tree_learning(clean_dataset, 0)
    dt_plot_clean = merge_keys(retrieve_tree(dt_clean))
    pt_clean = PlotTree(dt_plot_clean, depth_1o)
    pt_clean.plot(dt_plot_clean, (0.5, 1.0), '')
    # plt.title("Initial tree with clean dataset")
    plt.show()

    # plot unpruned tree with clean dataset
    plt.figure()
    (clean_split_test, clean_split_train) = k_fold_split(clean_dataset, 10, 3)
    (dt_unpruned_clean, depth_1u) = decision_tree.decision_tree_learning(clean_split_train, 0)
    dt_plot_clean_unpruned = merge_keys(retrieve_tree(dt_unpruned_clean))
    pt_pruned_clean = PlotTree(dt_plot_clean_unpruned, depth_1u)
    pt_pruned_clean.plot(dt_plot_clean_unpruned, (0.5, 1.0), '')
    # plt.title("Unpruned tree with clean dataset")
    plt.show()


    # plot pruned tree with clean dataset
    plt.figure()
    (clean_split_test, clean_split_train) = k_fold_split(clean_dataset, 10, 3)
    (dt_unpruned_clean, depth_11u) = decision_tree.decision_tree_learning(clean_split_train, 0)
    (dt_pruned_clean, depth_11p) = prune_tree(dt_unpruned_clean, clean_split_test)
    dt_plot_clean_pruned = merge_keys(retrieve_tree(dt_pruned_clean))
    pt_pruned_clean = PlotTree(dt_plot_clean_pruned, depth_11p)
    pt_pruned_clean.plot(dt_plot_clean_pruned, (0.5, 1.0), '')
    # plt.title("Pruned tree with clean dataset")
    plt.show()


    # plot initial tree with noisy dataset
    plt.figure()
    (dt_noisy, depth_2o) = decision_tree.decision_tree_learning(noisy_dataset, 0)
    dt_plot_noisy = merge_keys(retrieve_tree(dt_noisy))
    pt_noisy = PlotTree(dt_plot_noisy, depth_2o)
    pt_noisy.plot(dt_plot_noisy, (0.5, 1.0), '')
    # plt.title("Initial tree with noisy dataset")
    plt.show()

    # plot unpruned tree with noisy dataset
    plt.figure()
    (noisy_split_test, noisy_split_train) = k_fold_split(noisy_dataset, 10, 3)
    (dt_unpruned_noisy, depth_2u) = decision_tree.decision_tree_learning(noisy_split_train, 0)
    dt_plot_noisy_unpruned = merge_keys(retrieve_tree(dt_unpruned_noisy))
    pt_pruned_noisy = PlotTree(dt_plot_noisy_unpruned, depth_2u)
    pt_pruned_noisy.plot(dt_plot_noisy_unpruned, (0.5, 1.0), '')
    # plt.title("Unpruned tree with noisy dataset")
    plt.show()

    # plot pruned tree with clean dataset
    plt.figure()
    (noisy_split_test, noisy_split_train) = k_fold_split(noisy_dataset, 10, 3)
    (dt_unpruned_noisy, depth_22u) = decision_tree.decision_tree_learning(noisy_split_train, 0)
    (dt_pruned_noisy, depth_22p) = prune_tree(dt_unpruned_noisy, noisy_split_test)
    dt_plot_noisy_pruned = merge_keys(retrieve_tree(dt_pruned_noisy))
    pt_pruned_noisy = PlotTree(dt_plot_noisy_pruned, depth_22p)
    pt_pruned_noisy.plot(dt_plot_noisy_pruned, (0.5, 1.0), '')
    # plt.title("Pruned tree with noisy dataset")
    plt.show()


if __name__ == "__main__":
    main()
