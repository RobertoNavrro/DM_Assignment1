import numpy as np
import tree.tree_node as t

def tree_grow(x ,y, nmin: int, minleaf: int, nfeat: int) -> t.TreeNode:
    # nmin: Number of observations it must have to allow for a split.
    # minleaf: Number observations required for a node to be a leaf.
    # nfeat denotes the number of features considered for each split.
    
    #the first node begins with all the data
    root = t.TreeNode(x,y)
    nodelist = [root]
    print(type(nodelist))
    while(nodelist):
      current_node = nodelist[0]
      nodelist.remove(current_node)
      if calculateImpurity(current_node) > 0 and current_node.observations.shape[0] >= nmin:
          for column in current_node.observations:
              c = current_node.observations[column].values
              #check if the column is binary 
          break
    return root
    
def tree_pred(x,tr: t.Tree):
    pass

def calculateSplit(current_node: t.TreeNode, column):
    print("Hello")
    if all(v==0 or v==1 for v in current_node.observations[column].values):
        pass
    else:
        pass
    current_node.left = None
    current_node.right = None
    
def calculateImpurity(current_node: t.TreeNode):
    # print(f"The values are {current_node.labels.sum()} and {current_node.labels.shape[0]}")
    p_0 = current_node.labels.sum()/current_node.labels.shape[0]
    assert p_0 <= 1, "p_0 could not be computed as a probability"
    g_index = p_0*(1-p_0)
    print(g_index)
    return g_index