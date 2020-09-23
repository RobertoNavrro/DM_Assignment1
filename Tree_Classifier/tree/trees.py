import numpy as np
import pandas as pd
import tree.tree_node as t
from typing import Tuple

class Split:
    def __init__(self, left: t.TreeNode, right: t.TreeNode, column: str, parent_impurity: float):
        self.left = left
        self.right = right
        self.column = column
        self.delta_impurity = parent_impurity
        
    def splitImpurity(self):
        l_impurity = calculateImpurity(self.left)
        r_impurity = calculateImpurity(self.right)
        l_num_obs = self.left.observations.shape[0]
        r_num_obs = self.right.observations.shape[0]
        
    def printSplit(self):
        print(self.column)
        self.left.printTree()
        self.right.printTree()

def tree_grow(x: pd.DataFrame, y: pd.DataFrame, nmin: int, minleaf: int, nfeat: int) -> t.TreeNode:
    # nmin: Number of observations it must have to allow for a split.
    # minleaf: Number observations required for a node to be a leaf.
    # nfeat denotes the number of features considered for each split.
    
    #the first node begins with all the data
    root = t.TreeNode(x,y,None)
    nodelist = [root]
    split_list = list()

    while(nodelist):
      current_node = nodelist[0]
      nodelist.remove(current_node)
      current_impurity = calculateImpurity(current_node)
      if current_impurity > 0 and current_node.observations.shape[0] >= nmin:
          for column in current_node.observations:
              #A list that contains Splits
              split_list.append(generateSplit(current_node, column, current_impurity))
          # best_split = selectSplit(split_list,current_impurity)
          return split_list
          break
    return root
    
def tree_pred(x: pd.DataFrame, tr: t.TreeNode):
    pass

# def selectSplit(candidate_splits: list()) ->: Tuple(t.TreeNode, t.TreeNode, str):
    

def generateSplit(current_node: t.TreeNode, column: str, parent_impurity: int) -> Split: #Tuple[t.TreeNode, t.TreeNode, str]:
    #initialize 4 lists, do not use ([],) * 4 pythonism!
    left_x = list()
    left_y = list()
    right_x = list()
    right_y = list()
    # Check whether the current column is a categorical attribute
    if all(v==0 or v==1 for v in current_node.observations[column].values):
        for x_val, y_val in zip(current_node.observations[column].values, current_node.labels.values):
            if y_val == 0:
                left_x.append(x_val)
                left_y.append(y_val)
            elif y_val == 1:
                right_x.append(x_val)
                right_y.append(y_val)
        return Split(t.TreeNode(left_x,left_y,None), t.TreeNode(right_x,right_y,None), column, parent_impurity)
    # Otherwise current column is a numeric attribute
    else:
        pass
    
def calculateImpurity(current_node: t.TreeNode) -> float:
    # print(f"The values are {current_node.labels.sum()} and {current_node.labels.shape[0]}")
    p_0 = current_node.labels.sum()/current_node.labels.shape[0]
    assert p_0 <= 1, "p_0 could not be computed as a probability"
    g_index = p_0*(1-p_0)
    print(g_index)
    return g_index