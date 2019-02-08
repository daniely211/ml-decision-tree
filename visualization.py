import matplotlib.pyplot as plt
import decision_tree
import numpy as np
from evaluation import prune_tree, k_fold_split

class PlotTree:

    def __init__(self, tree, depth):
        self.axes = plt.subplot(111, frameon=False, **dict(xticks=[],yticks=[]))
        self.depth = float(depth)
        self.width = self.get_width(tree)
        self.x_margin = -1.0/self.width
        self.y_margin = 1.0


    def plot_node(self, text, loc, parent, node_type):
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
        centrePt = (self.x_margin + x_replacement , self.y_margin)
        decision_node = dict(boxstyle = "round4", fc = "w", color = 'dodgerblue')
        self.plot_node(root_decision, centrePt, parent, decision_node)

        self.y_margin -= 1.0/self.depth

        for key in ['left', 'right']:
            try:
                # if it is not a leaf node
                self.plot(node[key], centrePt, str(key))
            except:
                # a leaf node
                self.x_margin += 1.0 / self.width
                leaf_node = dict(boxstyle = "round4", fc = "w", color = 'dodgerblue')
                self.plot_node(node[key], (self.x_margin, self.y_margin), centrePt, leaf_node)

        self.y_margin += 1.0/self.depth


def retrieve_tree(tree):
    if not tree:
        return None

    left = merge_keys(tree['left'])
    right = merge_keys(tree['right'])
    tree['left'] = left
    tree['right'] = right
    if type(left).__name__ == 'dict':
        retrieve_tree(list(left.values())[0])
    if type(right).__name__ == 'dict':
        retrieve_tree(list(right.values())[0])

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
    clean_dataset = np.loadtxt('wifi_db/clean_dataset.txt')
    noisy_dataset = np.loadtxt('wifi_db/noisy_dataset.txt')

    (dt, depth_1) = decision_tree.decision_tree_learning(clean_dataset, 0)
    dt_plot = merge_keys(retrieve_tree(dt))
    pt = PlotTree(dt_plot, depth_1)
    pt.plot(dt_plot, (0.5, 1.0), '')

    plt.show()



if __name__ == "__main__":
    main()
