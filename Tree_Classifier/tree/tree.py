from __future__ import annotations
import numpy as np
from typing import List
import random
from collections import Counter


#Authors : Roberto Navarro - 4826260, Maria Galanty - 8118875, Giacomo Fiorentini - 4861310.



class TreeNode:
    """
    Class that allows us to represent a collection of lists as a tree node
    and inherently as a tree. A tree is composed of connected tree nodes.
    """
    def __init__(self, x: np.array, y: np.array, column: int):
        """
        Initializer function that allows the instantiation of the object tree.
        :param x: a 2D numpy array that contains the instances of the data.
        :param y: a 1D numpy array that contains the true labels of the data.
        :param column: an integer that indicates which column is used to split the tree node.
        :return: None
        """
        self.left = None
        self.right = None
        self.observations = x
        self.labels = y
        self.column = column
        self.split_value = None
        self.impurity = self.calculateNodeImpurity()

    def calculateNodeImpurity(self) -> float:
        """
        Function used to calculate the gini-index impurity of the node.
        """
        _sum = float(0)
        for v in self.labels:
            if v == 0:
                _sum += 1
        p_0 = _sum/self.labels.shape[0]
        gini_index = p_0 * (1-p_0)
        return gini_index

def tree_grow(x: np.array, y: np.array, nmin: int, minleaf: int, nfeat: int)-> TreeNode:
    """
    Function that creates a binary classification tree.
    :param x: a 2D numpy array that contains the instances of the data.
    :param y: a 1D numpy array that contains the true labels of the data.
    :param nmin: an integer that indicates the minimum number of observations required for a split to occur.
    :param minleaf: an integer that indicates the minimum number of observations required for a tree leaf.
    :param nfeat: an integer indicating the number of features that can be used to find a split.
    :return: the root of a Tree
    """
    root = TreeNode(x,y,None)
    nodelist = [root]
    while(nodelist):
        current_node = nodelist.pop(0)
        if current_node.impurity > 0:
            if nmin <= current_node.observations.shape[0]:
                splits = produceSplits(current_node, minleaf, nfeat)
                best_split = selectSplit(splits)
                if best_split:
                    applySplit(current_node, best_split)
                    nodelist.append(current_node.left)
                    nodelist.append(current_node.right)
    return root

def tree_grow_limited(x: np.array, y: np.array, nmin: int, minleaf: int, nfeat: int, limit: int)-> TreeNode:
    """
    Function that creates a binary classification tree.
    :param x: a 2D numpy array that contains the instances of the data.
    :param y: a 1D numpy array that contains the true labels of the data.
    :param nmin: an integer that indicates the minimum number of observations required for a split to occur.
    :param minleaf: an integer that indicates the minimum number of observations required for a tree leaf.
    :param nfeat: an integer indicating the number of features that can be used to find a split.
    :return: the root of a Tree
    """
    root = TreeNode(x,y,None)
    nodelist = [root]
    while(nodelist):
        current_node = nodelist.pop(0)
        if current_node.impurity > 0:
            if nmin <= current_node.observations.shape[0]:
                splits = produceSplits(current_node, minleaf, nfeat)
                best_split = selectSplit(splits)
                if best_split:
                    if(limit == 0):
                        break
                    limit = limit - 1
                    applySplit(current_node, best_split)
                    nodelist.append(current_node.left)
                    nodelist.append(current_node.right)

    return root


def tree_pred(x: np.array, tr: TreeNode) -> np.array:
    """
    Function responsible for obtaining the predictions of a tree, given a set of observations.
    :param x: a 2D numpy array that contains the instances of the data to be predicted.
    :param tr: a classification tree, trained on related data.
    :return: a 1D numpy array that contains the predicted classifications for the observations provided.
    """
    root = tr
    predicted_labels = []
    for row in range(x.shape[0]):
        while(tr):
            if type(tr.column) is not type(None):
                item = float(x.item((row,tr.column)))
                if item <= tr.split_value:
                    tr = tr.left
                elif item > tr.split_value: 
                    tr = tr.right
            else:
                predicted_labels.append(np.bincount(tr.labels).argmax())
                tr = None
        tr = root
    return np.asarray(predicted_labels)


def tree_grow_b(x: np.array, y: np.array, nmin: int, minleaf: int, nfeat: int, m: int)-> List[TreeNode] :
    """
    Function responsible for growing m trees, where every nodes is built based on nfeat random attributs
    :param x: a 2D numpy array that contains the instances of the data to be predicted.
    :param y: a 1D numpy array that contains the true labels of the data.
    :param nmin: an integer that indicates the minimum number of observations required for a split to occur.
    :param minleaf: an integer that indicates the minimum number of observations required for a tree leaf.
    :param nfeat: an integer indicating the number of features that can be used to find a split.
    :param m: an integer indicating the number of bootstrap samples to be drawn
    :return: a 1D array that contains the roots of all grown trees
    """
    tree_list = []
    for i in range (m):
        idx = np.random.randint(len(x), size=x.shape[0])
        samples =  x[idx, :]
        sampleslabels =  y[idx]
        
        root = TreeNode(samples,sampleslabels,None)
        nodelist = [root]
        while(nodelist):
            current_node = nodelist.pop(0)
            if current_node.impurity > 0:
                if nmin <= current_node.observations.shape[0]:
                    splits = produceSplits(current_node, minleaf, nfeat)
                    best_split = selectSplit(splits)
                    if best_split:
                        applySplit(current_node, best_split)
                        nodelist.append(current_node.left)
                        nodelist.append(current_node.right)
        tree_list.append(root)
    return tree_list

def tree_pred_b(x: np.array, tree_list: List[TreeNode]) -> np.array:
    """
    Function responsible for obtaining the prediction for an observation given a list of trees, for any number of observations.
    :param x: a 2D numpy array that contains the instances of the data to be predicted.
    :param tree_list: a list of classification trees, trained on related data.
    :return: a 1D numpy array that contains the predicted classifications, obtained from a majority decision given the output of every given tree, for the observations provided.
    """
    final_predicted_labels = []
    iter_predicted_labels = []
    for row in range(x.shape[0]):
        for tr in tree_list:
            while(tr):
                if type(tr.column) is not type(None):
                    item = float(x.item((row,tr.column)))
                    if item <= tr.split_value:
                        tr = tr.left
                    elif item > tr.split_value: 
                        tr = tr.right
                else:
                    iter_predicted_labels.append(np.bincount(tr.labels).argmax())
                    tr = None
        final_predicted_labels.append(np.bincount(iter_predicted_labels).argmax())
        iter_predicted_labels.clear()
    return np.asarray(final_predicted_labels)

class Split:
    """
    Class that allows us to represent a split for a tree node.
    """
    def __init__(self, left: TreeNode, right: TreeNode, column: int, split_value: float, parent_impurity: float):
        """
        Initializer function that allows the instantiation of a split.
        :param left: a tree node that is the left child of a tree node.
        :param right: a tree node that is the right child of a tree node.
        :param column: an integer that indicates which column is used to split the tree node.
        :param split_value: the value used to identify whether an observation goes to the left or right child.
        :param parent_impurity: the gini-index of the parent node of the left and right tree nodes.
        :return: None
        """
        self.left = left
        self.right = right
        self.column = column
        self.value = split_value
        self.delta_impurity = self.calculateDeltaImpurity(parent_impurity)
   
        
    def calculateDeltaImpurity(self, parent_impurity: float) -> float:
        """
        Function that allows us to calculate the delta impurity of a split.
        :param parent_impurity: the gini-index of the parent node that is being split.
        :return: a float that represents the delta impurity of the split.
        """
        l_impurity = self.left.impurity
        r_impurity = self.right.impurity
        l_num_obs = self.left.observations.shape[0]
        r_num_obs = self.right.observations.shape[0]
        obs_count = l_num_obs + r_num_obs    
        return parent_impurity - (l_num_obs/obs_count)*(l_impurity) - (r_num_obs/obs_count)*(r_impurity)


def binarySplit(current_node: TreeNode, split_attribute: int) -> bool:
    """
    Function checks whether the split is occuring on a binary attribute.
    :param current_node: the tree node that is to be split. 
    :param split_attribute: an integer that indicates which column is used to split the tree node.
    :return: a boolean that indicates whether the split is binary.
    """
    if all ((v==0 or v==1) for v in current_node.observations[:,split_attribute]):
        return True
    return False


def produceSplits(current_node: TreeNode, minleaf: int, nfeat: int) -> List[Split]:
    """
    Function that indentifies all possible splits randomly choosen nfeat attributes of a tree node.
    :param current_node: the tree node for which the splits are being calculated.
    :param minleaf: an integer that indicates the minimum number of observations required for a tree leaf.
    :param nfeat: an integer indicating the number of features that can be used to find a split.
    :return: a list of every possible split for a tree node.
    """
    random_columns = random.sample(range(0, (current_node.observations.shape[1])), nfeat)
    split_list = []
    for split_attribute in random_columns:
        if binarySplit(current_node,split_attribute):
            value_splitpoints = np.asarray([0])
        else:
            sorted_observation_values = np.sort(np.unique(current_node.observations[:,split_attribute]))
            value_splitpoints = (sorted_observation_values[0:sorted_observation_values.shape[0]-1] 
                                 + sorted_observation_values[1:sorted_observation_values.shape[0]])/2
        splits = generateSplits(current_node,split_attribute,minleaf,value_splitpoints)
        for spl_ in splits:
            split_list.append(spl_)
    return split_list


def applySplit(current_node: TreeNode, split: Split) -> None:
    """
    Function that applies a split to a tree node.
    :param current_node: the tree node for which the split is occurring.
    :param split: the split that is to be applied.
    :return: None
    """
    current_node.left = split.left
    current_node.right = split.right
    current_node.column = split.column
    current_node.split_value = split.value


def selectSplit(candidate_splits: List[Split]) -> Split:
    """
    Function that applies a split to a tree node.
    :param candidate_splits: a list of potential splits for a tree node.
    :param split: the split that is to be applied.
    :return: the best split possible.
    """
    delta_impurity = float(-1)
    best_split = None
    for split in candidate_splits:
        if split.delta_impurity > delta_impurity:
            best_split = split
            delta_impurity = split.delta_impurity
    return best_split


def generateSplits(current_node: TreeNode, column: int, minleaf: int, values: np.array) -> List:
    """
    Function responsible for generating a list of possible splits given a tree node.
    :param current_node: the tree node for which the splits are being generated for.
    :param column: an integer that indicates the attribute that the split will occur on.
    :param minleaf: an integer that indicates the minimum number of observations required for a tree leaf.
    :param values: a 1D numpy array that indicates the number of 
    :return: a list that contains all the possible splits given an attribute in the observations of a tree node.
    """
    left_x, left_y, right_x, right_y, split_list = [], [], [], [], []
    for split_value in values:
        for y_val, i in zip(current_node.labels, range(current_node.observations.shape[0])):
            obs_value = current_node.observations[i, column]
            if obs_value <= split_value:
                left_x.append(current_node.observations[i])
                left_y.append(y_val)
            elif obs_value > split_value:
                right_x.append(current_node.observations[i])
                right_y.append(y_val)
        if minLeafConstraint(left_y, right_y, minleaf):
            split_list.append(Split(TreeNode(np.asarray(left_x),np.asarray(left_y),None), TreeNode(np.asarray(right_x),np.asarray(right_y),None), column, split_value, current_node.impurity))
        left_x.clear()
        left_y.clear()
        right_x.clear()
        right_y.clear()
    return split_list


def minLeafConstraint(left_obs: List, right_obs: List, minleaf: int) -> bool:
    """
    Function responsible for checking the minLeafConstraint.
    :param left_obs: a list that contains the observations of a left child node.
    :param right_obs: a list that contains the observations of a right child node
    :return: a boolean indicating whether the constrain is met or not.
    """
    if len(left_obs) < minleaf or len(right_obs) < minleaf:
        return False
    else:
        return True
