import numpy as np

class TreeNode:
    def __init__(self, x: np.array, y: np.array, column: int):
        self.left = None
        self.right = None
        self.observations = x
        self.labels = y
        self.column = column

    def printNode(self):
        print(f"{self.column}:")
        print(f"{self.observations}")
        print(f"{self.labels}")

# Debugging function
def printTree(self, node: TreeNode):
    if node.left is not None:
        self.printTree(node.left)
       
    node.printNode()
        
    if node.right is not None:
        self.printTree(node.right)