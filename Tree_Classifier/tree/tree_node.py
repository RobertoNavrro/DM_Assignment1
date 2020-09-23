import pandas as pd
class TreeNode:
    def __init__(self, x: pd.DataFrame, y: pd.DataFrame, column: str):
        self.left = None
        self.right = None
        self.observations = x
        self.labels = y
        self.split = column
        
    def printTree(self):
        print(f"{self.observations}")
        print(f"{self.labels}")