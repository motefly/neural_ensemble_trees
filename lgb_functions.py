# -*- coding: utf-8 -*-
"""
Created on Wed Sept 12 14:37:22 2018
@author: xuzhenhui@pku.edu.cn
"""

import numpy as np
import lightgbm as lgb
import pdb

def CountSplitNodes(tree):
    root = tree['tree_structure']
    def counter(root):
        if 'split_index' not in root:
            return 0
        return 1 + counter(root['left_child']) + counter(root['right_child'])
    ans = counter(root)
    return ans

def GetItemByTree(tree, item='split_feature'):
    root = tree.raw['tree_structure']
    split_nodes = tree.split_nodes
    res = np.zeros(split_nodes+tree.raw['num_leaves'], dtype=np.int8)
    if 'value' in item:
        res = res.astype(np.float64)
    def getFeature(root, res):
        if 'child' in item:
            if 'split_index' in root:
                node = root[item]
                if 'split_index' in node:
                    res[root['split_index']] = node['split_index']
                else:
                    res[root['split_index']] = node['leaf_index'] + split_nodes # need to check
            else:
                res[root['leaf_index'] + split_nodes] = -1
        elif 'value' in item:
            if 'split_index' in root:
                res[root['split_index']] = root['internal_'+item]
            else:
                res[root['leaf_index'] + split_nodes] = root['leaf_'+item]
        else:
            if 'split_index' in root:
                res[root['split_index']] = root[item]
            else:
                res[root['leaf_index'] + split_nodes] = -2
        if 'left_child' in root:
            getFeature(root['left_child'], res)
        if 'right_child' in root:
            getFeature(root['right_child'], res)
    getFeature(root, res)
    return res

def GetTreeSplits(model):
    # model = gbm.dump_model()
    featurelist = []
    threhlist = []
    trees = []
    for idx, tree in enumerate(model['tree_info']):
        trees.append(treeInterpreter(tree))
        featurelist.append(trees[-1].feature)
        threhlist.append(GetItemByTree(trees[-1], 'threshold'))
    return (trees, featurelist, threhlist)


def GetChildren(trees):
    listcl = []
    listcr = []
    for idx, tree in enumerate(trees):
        listcl.append(GetItemByTree(tree, 'left_child'))
        listcr.append(GetItemByTree(tree, 'right_child'))
    return(listcl, listcr)

def GetTreePaths(trees):
    pass

# def GetLeafValue(tree):
    

class treeInterpreter(object):
    def __init__(self, tree):
        self.raw = tree
        self.split_nodes = CountSplitNodes(tree)
        self.node_count = self.split_nodes + tree['num_leaves']
        self.value = GetItemByTree(self, item='value')
        self.feature = GetItemByTree(self)
        # self.leaf_value = GetLeafValue(tree)
