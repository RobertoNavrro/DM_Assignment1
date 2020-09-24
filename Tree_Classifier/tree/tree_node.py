import numpy as np

class TreeNode:
    def __init__(self, x: np.array, y: np.array, column: int):
        self.left = None
        self.right = None
        self.observations = x
        self.labels = y
        self.column = column
    
    def set_column(self,column):
        self.column= column
    
    def set_left(self, left_child):
        self.left = left_child
    
    def set_right(self, right_child):
        self.right = right_child
    
    def printNode(self):
        print(f"{self.column}:")
        print(f"{self.observations}")
        print(f"{self.labels}")

def printTree(self, node: TreeNode):
    if node.left is not None:
        self.printTree(node.left)
       
    node.printNode()
        
    if node.right is not None:
        self.printTree(node.right)