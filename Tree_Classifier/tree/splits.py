import tree.tree_node as t
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
    
    # Debugging function    
    def printSplit(self):
        print(f"Column:{self.column}, DeltaImpurity: {self.delta_impurity}")
        print("================")
        self.left.printNode()
        print("----------------")
        self.right.printNode()
        print("================")
        
        
        
# Functions that are not part of the split class itself,
# but provide functionality &/or create a Split

      
def applySplit(split: Split, tree_node: t.TreeNode) -> None:
    tree_node.left = split.left
    tree_node.right = split.right
    tree_node.column = split.column

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

def binarySplit(current_node: t.TreeNode, column: int, parent_impurity: float, minleaf: int) -> Split:
    #initialize 4 lists, do not use ([],) * 4 pythonism! it messes with the list allocation
    left_x = list()
    left_y = list()
    right_x = list()
    right_y = list()
    for x_val, y_val, i in zip(current_node.observations[:, column], current_node.labels, range(current_node.observations.shape[0])):
        # Fill in the observations that allow the split, we are storing the entire row!
        if x_val == 0:
            # To prevent messing with different tree values, we make copies (More memory but who cares)
            left_x.append(current_node.observations[i])
            left_y.append(y_val)
        elif x_val == 1:
            right_x.append(current_node.observations[i])
            right_y.append(y_val)
                
        # If our current split has a leaf that does not meet the minleaf constraint we do not allow it.
    if not minLeafConstraint(left_x, right_x, minleaf):
        return None
    else:
        return Split(t.TreeNode(np.asarray(left_x),np.asarray(left_y),None,current_node.node_id), t.TreeNode(np.asarray(right_x),np.asarray(right_y),None,current_node.node_id), column, parent_impurity)
    
def numericalSplit(current_node: t.TreeNode, column: int, parent_impurity: float, minleaf: int) -> Split:
    #initialize 4 lists, do not use ([],) * 4 pythonism! it messes with the list allocation
    left_x = list()
    left_y = list()
    right_x = list()
    right_y = list()
    
    obs_copy = current_node.observations.copy()
    label_copy = current_node.labels.copy().transpose()
    
    # Combine the Observations and copies to prevent messing up labels
    full_stack = np.insert(obs_copy,obs_copy.shape[1],label_copy,axis=1)
    
    # Sort the observations+labels based on a column
    full_stack = full_stack[full_stack[:,column].argsort()]
    
    
    
    
    
    print(full_stack)
    print("\n\n")
    
    return None
    

def calculateNodeImpurity(current_node: t.TreeNode) -> float:
    _sum = float(0)
    # We can easily see how many 1s exist by adding them together
    for v in current_node.labels:
        _sum += v
    p_0 = _sum/current_node.labels.shape[0]
    gini_index = p_0*(1-p_0)
    return gini_index