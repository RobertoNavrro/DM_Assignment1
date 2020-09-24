import tree.tree_node as t
import pandas as pd
import numpy as np

class Split:
    def __init__(self, left: t.TreeNode, right: t.TreeNode, column: int, parent_impurity: float):
        self.left = left
        self.right = right
        self.column = column
        self.delta_impurity = self.calculateDeltaImpurity(parent_impurity)
        
    def calculateDeltaImpurity(self, parent_impurity: float):
        l_impurity = calculateNodeImpurity(self.left)
        r_impurity = calculateNodeImpurity(self.right)
        l_num_obs = self.left.observations.shape[0]
        r_num_obs = self.right.observations.shape[0]
        obs_count = l_num_obs + r_num_obs
        return parent_impurity - (l_num_obs/obs_count)*(l_impurity) - (r_num_obs/obs_count)*(r_impurity)
        
    def printSplit(self):
        print(f"Column:{self.column}, DeltaImpurity: {self.delta_impurity}")
        print("================")
        self.left.printNode()
        print("----------------")
        self.right.printNode()
        print("================")
        
        
        
# Functions that are not part of the split class itself,
# but provide functionality &/or create a Split

      
def applySplit(split: Split, tree_node: t.TreeNode):
    tree_node.set_left(split.left)
    tree_node.set_right(split.right)
    tree_node.set_column(split.column)     

def selectSplit(candidate_splits: list()) -> Split:
    delta_impurity = float(0)
    best_split = None
    for split in candidate_splits:
        if split.delta_impurity > delta_impurity:
            best_split = split
            delta_impurity = split.delta_impurity
    return best_split

def generateSplit(current_node: t.TreeNode, column: int, parent_impurity: float, minleaf: int) -> Split:
    #initialize 4 lists, do not use ([],) * 4 pythonism! it messes with the list allocation
    left_x = list()
    left_y = list()
    right_x = list()
    right_y = list()
    # Check whether the current column is a categorical attribute
    if all(v==0 or v==1 for v in current_node.observations[ :, column]):
        for x_val, y_val, i in zip(current_node.observations[:, column], current_node.labels, range(current_node.observations.shape[0])):
            # Fill in the observations that allow the split
            if x_val == 0:
                left_x.append(current_node.observations[i].copy())
                left_y.append(y_val)
            elif x_val == 1:
                right_x.append(current_node.observations[i].copy())
                right_y.append(y_val)
                
        # If our current split has a leaf that does not meet the minleaf constraint we do not allow it.
        if len(left_x) < minleaf or len(right_x) < minleaf:
            return None
        else:
            return Split(t.TreeNode(np.asarray(left_x),np.asarray(left_y),None), t.TreeNode(np.asarray(right_x),np.asarray(right_y),None), column, parent_impurity)
    # Otherwise current column is a numeric attribute, find best numeric
    else:
        
        return None

def calculateNodeImpurity(current_node: t.TreeNode) -> float:
    # print(f"The values are {current_node.labels.sum()} and {current_node.labels.shape[0]}")
    _sum = float(0)
    for v in current_node.labels:
        _sum += v
        
    p_0 = _sum/current_node.labels.shape[0]
    g_index = p_0*(1-p_0)
    return g_index