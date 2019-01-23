# import numpy as np
# clean_dataset = np.loadtxt('wifi_db/clean_dataset.txt')
# noisy_dataset = np.loadtxt('wifi_db/noisy_dataset.txt')
#
# def entropy(dataset):
#     allLabel = np.array(dataset[:,-1])
#     total_num = dataset.shape[0]
#     probLabel = np.zeros(4)
#     H = 0
#
#     for i in range(0,4):
#         probLabel[i] = np.count_nonzero(allLabel == i+1)/total_num
#         print(probLabel[i])
#         H = H - probLabel[i] * np.log2(probLabel[i])
#
#     return H
#
# #print(entropy(clean_dataset))
#
# def remainder(sLeft, sRight):
#     numLeft = np.shape(sLeft)[0]
#     numRight = np.shape(sRight)[0]
#
#     r = numLeft / (numLeft + numRight) * entropy(sLeft) + numRight / (numLeft + numRight) * entropy(sRight)# remainder
#
#     return r

#print(remainder(clean_dataset))


import sklearn.datasets as datasets
import pandas as pd
iris=datasets.load_iris()
df=pd.DataFrame(iris.data, columns=iris.feature_names)
y=iris.target

from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(df,y)

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()

export_graphviz(dtree, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
