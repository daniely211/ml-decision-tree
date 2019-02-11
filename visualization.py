import matplotlib.pyplot as plt
import decision_tree
import numpy as np
import sys
from evaluation import prune_tree, k_fold_split
from numpy.random import shuffle

class PlotTree:

    def __init__(self, tree, depth):
        self.axes = plt.subplot(111, frameon=False, **dict(xticks=[],yticks=[]))
        self.depth = float(depth)
        self.width = self.get_width(tree)
        self.x_loc = -1.0/self.width
        self.y_loc = 1.0

    # Plot the node on a diagram
    #
    #   text        the text within the node
    #   loc         location of node text (x,y)
    #   xy          the location of the node
    #   node_type   style of the box
    def plot_node(self, text, loc, xy, node_type):
        args = dict(arrowstyle = "-", color = 'gold', connectionstyle = "arc3")
        self.axes.annotate(text, xy = xy, xycoords = 'axes fraction', xytext = loc, textcoords = 'axes fraction',
                           va = 'bottom', ha = 'center', bbox = node_type, arrowprops = args)

    # Get the width of the tree (number of leafs)
    #
    #   tree    the tree
    def get_width(self, tree):
        num_leaf_nodes = 0

        root_decision = list(tree.keys())[0]
        node = tree[root_decision]

        for key in ['left', 'right']:
            try:
                num_leaf_nodes += self.get_width(node[key])
            except:
                num_leaf_nodes += 1

        return num_leaf_nodes

    # Plot the tree
    #
    #   tree    the tree to be plot
    #   xy      the coordinates of the node
    #   text    the text within the node
    def plot(self, tree, xy, text = ''):
        num_leaf_nodes = self.get_width(tree)

        root_decision = list(tree.keys())[0]
        node = tree[root_decision]

        # Plot the tree in the center
        x_replacement = (1.0 + float(num_leaf_nodes)) / 2.0 / self.width
        centre_pt = (self.x_loc + x_replacement , self.y_loc)
        decision_node = dict(boxstyle = "round4", fc = "w", color = 'dodgerblue')
        self.plot_node(root_decision, centre_pt, xy, decision_node)

        self.y_loc -= 1.0/self.depth

        for key in ['left', 'right']:
            try:
                # if it is not a leaf node
                self.plot(node[key], centre_pt, str(key))
            except:
                # a leaf node
                self.x_loc += 1.0 / self.width
                leaf_node = dict(boxstyle = "round4", fc = "w", color = 'green')
                self.plot_node(node[key], (self.x_loc, self.y_loc), centre_pt, leaf_node)

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

# Convert the tree into the format we use in PlotTree
#
#   tree    the tree to plot
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

# Plot the trees of all different formats
def main(dataset_path):
    # upload clean and noisy dataset
    dataset = np.loadtxt(dataset_path)
    shuffle(dataset)

    # plot unpruned tree
    plt.figure()
    (clean_split_test, clean_split_train) = k_fold_split(dataset, 10, 3)
    (dt_unpruned_clean, depth_1u) = decision_tree.decision_tree_learning(clean_split_train, 0)
    dt_plot_clean_unpruned = merge_keys(retrieve_tree(dt_unpruned_clean))
    pt_pruned_clean = PlotTree(dt_plot_clean_unpruned, depth_1u)
    pt_pruned_clean.plot(dt_plot_clean_unpruned, (0.5, 1.0))
    plt.show()

    # plot pruned tree
    plt.figure()
    (clean_split_test, clean_split_train) = k_fold_split(dataset, 10, 3)
    (dt_unpruned_clean, depth_11u) = decision_tree.decision_tree_learning(clean_split_train, 0)
    (dt_pruned_clean, depth_11p) = prune_tree(dt_unpruned_clean, clean_split_test)
    dt_plot_clean_pruned = merge_keys(retrieve_tree(dt_pruned_clean))
    pt_pruned_clean = PlotTree(dt_plot_clean_pruned, depth_11p)
    pt_pruned_clean.plot(dt_plot_clean_pruned, (0.5, 1.0))
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: please pass the data set to be evaluated as an argument e.g.:")
        print()
        print("$  python3 visualisation.py wifi_db/noisy_dataset.txt")
        print()
        sys.exit()
    else:
        main(sys.argv[1])
