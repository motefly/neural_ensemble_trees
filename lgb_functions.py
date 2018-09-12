# -*- coding: utf-8 -*-
"""
Created on Wed Sept 12 14:37:22 2018
@author: xuzhenhui@pku.edu.cn
"""

import numpy as np
import lightgbm as lgb

def GetItemByTree(tree, item='feature'):
    res_dict = {}
    root = tree['tree_structure']
    def getFeature(root, res_dict):
        if 'split_index' not in root:
            return
        if 'child' in item:
            node = root[item]
            if 'split_index' in node:
                res_dict[root['split_index']] = node['split_index']
            else:
                res_dict[root['split_index']] = -2 # need to check
        else:
            res_dict[root['split_index']] = root['split_'+item]
        getFeature(root['left_child'], res_dict)
        getFeature(root['right_child'], res_dict)
    getFeature(root, res_dict)
    res = []
    cur = 0
    for k,v in res_dict.items():
        if k == cur:
            res.append(v)
            cur += 1
        else:
            while k!=cur:
                res.append(-2)
                cur += 1
    return res

def GetTreeSplits(gbm):
    model = gbm.dump_model()
    featurelist = []
    threhlist = []
    trees = model['tree_info']
    for tree, idx in enumerate(trees):
        featurelist.append(GetItemByTree(tree))
        threhlist.append(GetItemByTree(tree, 'threshold'))
    return (trees, featurelist, threshlist)


def GetChildren(trees):
    listcl = []
    listcr = []
    for tree, idx in enumerate(trees):
        listcl.append(GetItemByTree(tree, 'left_child'))
        listcr.append(GetItemByTree(tree, 'right_child'))
    return(listcl, listcr)


"""
Have not update for lightGBM
def GetActiveNodes(sample,featurelist, threshlist, trees):
    #returns a list over all trees with all corresponding active nodes
    #(featurelist, threshlist, trees) = GetTreeSplits(rf)

    act_node_list=[]      #list, containing all reached nodes for each tree listed

    for i in range(len(featurelist)):
        actives = []
        currentnode = 0
        actives.append(0)

        while featurelist[i][currentnode] != -2:    #not at end leaf yet

            if sample[featurelist[i][currentnode]] >= threshlist[i][currentnode]:
                currentnode = trees[i].children_right[currentnode]
            else:
                currentnode = trees[i].children_left[currentnode]

            actives.append(currentnode)

        act_node_list.append(actives)

    return act_node_list


def LASSO_Input(data,rf):
    #Returns one big indicator vector. labels are not needed.


    (trees, featurelist, threshlist) = GetTreeSplits(rf)

    act_node_list = GetActiveNodes(data[0,:],featurelist, threshlist, trees)

    LassoData2Trees = []
    ls=[]
    for i in range(rf.n_estimators):                        #over all trees
        for j in range(rf.estimators_[i].tree_.node_count): #over all nodes
            ls.append(int( j in act_node_list[i] ) )#if active node, 1, else, 0
            LassoData2Trees.append([i,j])

    IndLen = len(ls)
    N = data.shape[0]
    LassoData = np.zeros([N, IndLen])


    for k in range(N):
        act_node_list = GetActiveNodes(data[k,:],featurelist, threshlist, trees)
        ls = []
        for i in range(rf.n_estimators):                        #over all trees
            for j in range(rf.estimators_[i].tree_.node_count): #over all nodes
                ls.append(int( j in act_node_list[i] ) )#if active node, 1, else, 0

        LassoData[k,:] = np.asarray(ls)
    return (LassoData, LassoData2Trees)



def node_indicator_function(sample, rf):
    #returns a vector of node activities over all tree nodes, over all trees.
    #The output is
    #zero for inactive nodes, and 1 for active nodes. In case of the queried
    #node being a leaf node, the predicted class is returned as well.

    (featurelist, threshlist) = GetTreeSplits(rf)
    (act_node_list, active_feats_list, active_threshs_list) = GetActiveNodes(sample, rf)

    label = 0
    leafind = False
    if act_node_list[treei].count(nodej) > 0:   #if active
        indicator = 1.0
        #print treei, nodej, len(act_node_list[treei])
        if act_node_list[treei][len(act_node_list[treei])-1]==nodej: #if end leaf
            label = rf.predict(sample)
            leafind = True

    else:
        indicator = 0.0

    return (indicator, label, leafind)



def Linear_Forest_Prediction(rf, sample):

    (featurelist, threshlist) = GetTreeSplits(rf)

    Linear_Pred = np.zeros([rf.n_classes_, 1])

    #over all trees and their nodes, get their contribution to the prediction.
    for i in range(rf.n_estimators):
        for j in range(np.sum(featurelist[i]!=-2)): #for all nodes j in tree i
            (indicator, label, leafind) = node_indicator_function(sample, rf, i, j)
            if leafind == True:
                Linear_Pred[int(label)] +=1  #add one to the scroe for the label predicted

    return Linear_Pred
"""
