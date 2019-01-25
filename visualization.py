import matplotlib.pyplot as plt
from ast import literal_eval
import decision_trees
import numpy as np

decisionNode=dict(boxstyle="round4",fc="w", color ='dodgerblue')
leafNode=dict(boxstyle="round4",fc="w", color='dodgerblue')
arrow_args=dict(arrowstyle="-",color='gold', connectionstyle="arc3")

def plotNode(nodeText,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeText, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction',
                           va='bottom', ha='center', bbox=nodeType, arrowprops=arrow_args)

def getNumLeafs(myTree):
    numLeafs = 0
    firstList = list(myTree.keys())
    firstStr=firstList[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs+=getNumLeafs(secondDict[key])
        else: numLeafs+=1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth=0
    firstList=list(myTree.keys())
    firstStr=firstList[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth=1+getTreeDepth(secondDict[key])
        else: thisDepth=1
        if thisDepth>maxDepth:
            maxDepth=thisDepth
    return maxDepth


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstList = list(myTree.keys())
    firstStr=firstList[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def createPlot(inTree):
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    axprops=dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW=float(getNumLeafs(inTree))
    plotTree.totalD=float(getTreeDepth(inTree))
    plotTree.xOff=-0.5/plotTree.totalW
    plotTree.yOff=1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()


def retrieveTree(tree):
    if not tree:
        return None

    left = Combine(tree.get('left'))
    right = Combine(tree.get('right'))
    tree['left'] = left;
    tree['right'] = right;
    if type(left).__name__ == 'dict':
        retrieveTree(list(left.values())[0])
    if type(right).__name__ == 'dict':
        retrieveTree(list(right.values())[0])
    return tree


def Combine(tree):
    leaf = tree.get('leaf');
    value = tree.pop('value')
    attribute = tree.pop('attribute')

    new = {}
    if leaf == None:
        condition = 'x'+str(attribute)+'<'+str(value)
        new[condition] = tree;
        return new
    else:
        return 'leaf:'+str(value)

# clean_dataset = np.loadtxt('wifi_db/clean_dataset.txt')
# dt = decision_trees.decision_tree_learning(clean_dataset, 0)
dt = literal_eval(open('cleantree.txt', 'r').read())[0]
# dt = literal_eval(open('noisytree.txt', 'r').read())[0]
decisionTree = Combine(retrieveTree(dt))
print(decisionTree)

createPlot(decisionTree)
# print(getTreeDepth(decisionTree))
