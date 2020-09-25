import tree.tree_node as t
import numpy as np
from tree.splits import calculateNodeImpurity, selectSplit, applySplit, generateSplits

def tree_grow(x: np.array, y: np.array, nmin: int, minleaf: int, nfeat: int) -> t.TreeNode:
    # nmin: Number of observations it must have to allow for a split.
    # minleaf: Number observations required for a node to be a leaf.
    # nfeat denotes the number of features considered for each split.
    
    #the first node begins with all the data
    root = t.TreeNode(x, y, None,-1)
    nodelist = [root]
    split_list = list()
    ID = 0
    while(nodelist):
      # Obtain the first item from the nodelist
      current_node = nodelist[0]
      nodelist.remove(current_node)
      current_impurity = calculateNodeImpurity(current_node)
      current_node.node_id = ID
      if current_impurity > 0 and current_node.observations.shape[0] >= nmin:
          for column in range(current_node.observations.shape[1]):
              if all(v==0 or v==1 for v in current_node.observations[ :, column]):
                  splits = generateSplits(current_node, column, current_impurity, minleaf, np.asarray([0]))
              else:
                  sorted_values = np.sort(np.unique(current_node.observations[:,column]))
                  value_splitpoints = (sorted_values[0:sorted_values.shape[0]-1] + sorted_values[1:sorted_values.shape[0]])/2
                  splits = generateSplits(current_node, column, current_impurity, minleaf, value_splitpoints)
              if splits is not None:
                  for split in splits:
                      split_list.append(split)               
          if len(split_list) > 0:
              applySplit(current_node, selectSplit(split_list))
              nodelist.append(current_node.left)
              nodelist.append(current_node.right)
              split_list.clear()
              ID+=1
              
    return root


def tree_pred(x: np.array, tr: t.TreeNode):
    
    for entry in x:
        pass
    
    pass
    
def printTree(node: t.TreeNode):
    if node.left is not None:
        printTree(node.left)
    print("\n")
    node.printNode()
    print("\n")     
    if node.right is not None:
        printTree(node.right)
    