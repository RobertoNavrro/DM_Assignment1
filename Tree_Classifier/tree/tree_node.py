import numpy as np

class TreeNode:
    def __init__(self, x: np.array, y: np.array, column: int):
        self.left = None
        self.right = None
        self.observations = x
        self.labels = y
        self.column = column
        self.split_value = None
        self.impurity = self.calculateNodeImpurity()

    def printNode(self):
        print(f"Split index:value = {self.column}:{self.split_value}")
        print(f"{self.observations} \n")
        print(f"{self.labels}")
    
    def calculateNodeImpurity(self) -> float:
        _sum = float(0)
        # We can easily see how many 1s exist by adding them together
        for v in self.labels:
            if v == 0:
                _sum += 1
        p_0 = _sum/self.labels.shape[0]
        gini_index = p_0 * (1-p_0)
        return gini_index