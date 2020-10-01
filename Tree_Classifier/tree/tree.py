import numpy as np
from typing import List

class TreeNode:
    def __init__(self, x: np.array, y: np.array, column: int):
        self.left = None
        self.right = None
        self.observations = x
        self.labels = y
        self.column = column
        self.split_value = None
        self.impurity = self.calculateNodeImpurity()

    def calculateNodeImpurity(self) -> float:
        _sum = float(0)
        # We can easily see how many 1s exist by adding them together
        for v in self.labels:
            if v == 0:
                _sum += 1
        p_0 = _sum/self.labels.shape[0]
        gini_index = p_0 * (1-p_0)
        return gini_index

def tree_grow(x: np.array, y: np.array, nmin: int, minleaf: int, nfeat: int)-> TreeNode:
    root = TreeNode(x,y,None)
    nodelist = [root]
    while(nodelist):
        current_node = nodelist.pop(0)
        if current_node.impurity > 0:
            if nmin <= current_node.observations.shape[0]:
                splits = produceSplits(current_node, minleaf)
                best_split = selectSplit(splits)
                if best_split:
                    applySplit(current_node, best_split)
                    nodelist.append(current_node.left)
                    nodelist.append(current_node.right)
    return root


def tree_pred(x: np.array, tr: TreeNode) -> np.array:
    root = tr
    predicted_labels = list()
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

class Split:
    def __init__(self, left: TreeNode, right: TreeNode, column: int, split_value: float, parent_impurity: float):
        self.left = left
        self.right = right
        self.column = column
        self.value = split_value
        self.delta_impurity = self.calculateDeltaImpurity(parent_impurity)
   
        
    def calculateDeltaImpurity(self, parent_impurity: float) -> float:
        l_impurity = self.left.impurity
        r_impurity = self.right.impurity
        l_num_obs = self.left.observations.shape[0]
        r_num_obs = self.right.observations.shape[0]
        obs_count = l_num_obs + r_num_obs    
        return parent_impurity - (l_num_obs/obs_count)*(l_impurity) - (r_num_obs/obs_count)*(r_impurity)


def binarySplit(current_node: TreeNode,split_attribute: int) -> bool:
    if all ((v==0 or v==1) for v in current_node.observations[:,split_attribute]):
        return True
    return False


def produceSplits(current_node: TreeNode, minleaf: int) -> List[Split]:
    split_list = []
    for split_attribute in range(current_node.observations.shape[1]):
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


def applySplit(tree_node: TreeNode, split: Split) -> None:
    tree_node.left = split.left
    tree_node.right = split.right
    tree_node.column = split.column
    tree_node.split_value = split.value


def selectSplit(candidate_splits: list()) -> Split:
    delta_impurity = float(-1)
    best_split = None
    for split in candidate_splits:
        if split.delta_impurity > delta_impurity:
            best_split = split
            delta_impurity = split.delta_impurity
    return best_split


def generateSplits(current_node: TreeNode, column: int, minleaf: int, values: np.array) -> list:
    left_x, left_y, right_x, right_y, split_list = [], [], [], [], []
    for split_value in values:
        for y_val, i in zip(current_node.labels, range(current_node.observations.shape[0])):
            # Fill in the observations that allow the split, we are storing the entire row!
            obs_value = current_node.observations[i, column]
            if obs_value <= split_value: # if less than 1 in binary, if less than x some value in numerical
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


def minLeafConstraint(left_obs: list, right_obs: list, minleaf: int) -> bool:
    if len(left_obs) < minleaf or len(right_obs) < minleaf:
        return False
    else:
        return True
