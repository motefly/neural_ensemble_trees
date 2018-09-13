
class modelInterpreter(object):
    def __init__(self, model, tree_model):
        print("Model Interpreting...")
        self.tree_model = tree_model
        if tree_model == 'randomforest':
            from forest_functions import GetTreeSplits, GetChildren
            self.n_features_ = model.n_features_
        else:
            model = model.dump_model()
            from lgb_functions import GetTreeSplits, GetChildren
            self.n_features_ = model['max_feature_idx'] + 1
        self.trees, self.featurelist, self.threshlist = GetTreeSplits(model)
        self.listcl, self.listcr = GetChildren(self.trees)

    def GetTreeSplits(self):
        return (self.trees, self.featurelist, self.threshlist)

    def GetChildren(self):
        return (self.listcl, self.listcr)
