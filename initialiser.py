from layer_initialisation import InitFirstLayer
from layer_initialisation import InitSecondLayer
from layer_initialisation import InitThirdLayer

class modelInterpreter(object):
    def __init__(self, model, tree_model):
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

def get_network_initialisation_parameters(model, strength01=100.0, strength12=1.0, tree_model='lightgbm'):
    """
    Given a pre-trained random forest model, this function returns as numpy arrays
    the weights and biases for initialising a 2-layer feedforward neural network.
    The strength01 and strength12 are hyperparameters that determine how strongly
    the continuous neural network nonlinearity will approximate a discrete step function
    """

    modelI = modelInterpreter(model, tree_model)

    # get network parameters for first hidden layer
    W1, b1, nodelist1 = InitFirstLayer(modelI, strength01)

    # get network parameters for second hidden layer
    W2, b2, leaf_neurons = InitSecondLayer(modelI, nodelist1, strength12)

    # get network parameters for third hidden layer
    W3 = InitThirdLayer(modelI, leaf_neurons)

    return W1, b1, W2, b2, W3

