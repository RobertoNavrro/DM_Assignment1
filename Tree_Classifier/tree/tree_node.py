import numpy as np

class TreeNode:
    def __init__(self, x: np.array, y: np.array, column: int, parent_id: int):
        self.left = None
        self.right = None
        self.observations = x
        self.labels = y
        self.column = column
        self.split_value = None
        self.node_id = None
        self.child_of = parent_id

    def printNode(self):
        print(f"Split index:value = {self.column}:{self.split_value} - Parent_Id = {self.child_of} - Node_Id = {self.node_id}")
        print(f"{self.observations} \n")
        print(f"{self.labels}")

# Debugging function
def printTree(self, node: TreeNode):
    if node.left is not None:
        self.printTree(node.left)
       
    node.printNode()
        
    if node.right is not None:
        self.printTree(node.right)