import tree.tree_node as t
import numpy as np
from tree.splits import calculateNodeImpurity, selectSplit, applySplit, binarySplit, numericalSplit

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
                  candidate = binarySplit(current_node, column, current_impurity, minleaf)
              else:
                  candidate = numericalSplit(current_node, column, current_impurity, minleaf)
                  candidate = None
              if candidate is not None:
                  split_list.append(candidate)
                  
          if len(split_list) > 0:
              applySplit(selectSplit(split_list),current_node)
              nodelist.append(current_node.left)
              nodelist.append(current_node.right)
              split_list.clear()
              ID+=1
              
    return root


def tree_pred(x: np.array, tr: t.TreeNode):
    pass
    
def printTree(node: t.TreeNode):
    if node.left is not None:
        printTree(node.left)
    print("\n")
    node.printNode()
    print("\n")     
    if node.right is not None:
        printTree(node.right)
    