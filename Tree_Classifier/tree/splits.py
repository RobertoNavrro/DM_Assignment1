import tree.tree_node as t
import numpy as np

class Split:
    def __init__(self, left: t.TreeNode, right: t.TreeNode, column: int, split_value: float, parent_impurity: float):
        self.left = left
        self.right = right
        self.column = column
        self.value = split_value
        self.delta_impurity = self.calculateDeltaImpurity(parent_impurity)
        
    def calculateDeltaImpurity(self, parent_impurity: float):
        l_impurity = calculateNodeImpurity(self.left)
        r_impurity = calculateNodeImpurity(self.right)
        l_num_obs = self.left.observations.shape[0]
        r_num_obs = self.right.observations.shape[0]
        obs_count = l_num_obs + r_num_obs
        # Sometimes this appears to be negative, this might be an issue
        return parent_impurity - (l_num_obs/obs_count)*(l_impurity) - (r_num_obs/obs_count)*(r_impurity)
        
# Functions that are not part of the split class itself,
# but provide functionality &/or create a Split
      
def applySplit(tree_node: t.TreeNode, split: Split) -> None:
    tree_node.left = split.left
    tree_node.right = split.right
    tree_node.column = split.column
    tree_node.split_value = split.value

def selectSplit(candidate_splits: list()) -> Split:
    delta_impurity = float(0)
    best_split = None
    for split in candidate_splits:
        if split.delta_impurity > delta_impurity:
            best_split = split
            delta_impurity = split.delta_impurity
    return best_split

def minLeafConstraint(left_obs: list, right_obs: list, minleaf: int) -> bool:
    if len(left_obs) < minleaf or len(right_obs) < minleaf:
        return False
    else:
        return True


def generateSplits(current_node: t.TreeNode, column: int, parent_impurity: float, minleaf: int, values: np.array) -> list:
    left_x = list()
    left_y = list()
    right_x = list()
    right_y = list()
    
    split_list = list() 
    for split_value in values:
        for x_val, y_val, i in zip(current_node.observations[:, column], current_node.labels, range(current_node.observations.shape[0])):
            # Fill in the observations that allow the split, we are storing the entire row!
            if x_val <= split_value: # if less than 1 in binary, if less than x some value in numerical
                left_x.append(current_node.observations[i])
                left_y.append(y_val)
            elif x_val > split_value:
                right_x.append(current_node.observations[i])
                right_y.append(y_val)
        if minLeafConstraint(left_x, right_x, minleaf):
            split_list.append(Split(t.TreeNode(np.asarray(left_x),np.asarray(left_y),None,current_node.node_id), t.TreeNode(np.asarray(right_x),np.asarray(right_y),None,current_node.node_id), column, split_value, parent_impurity))
        left_x.clear()
        left_y.clear()
        right_x.clear()
        right_y.clear()
        
    return split_list


def calculateNodeImpurity(current_node: t.TreeNode) -> float:
    _sum = float(0)
    # We can easily see how many 1s exist by adding them together
    for v in current_node.labels:
        _sum += v
    p_0 = _sum/current_node.labels.shape[0]
    gini_index = p_0*(1-p_0)
    return gini_index