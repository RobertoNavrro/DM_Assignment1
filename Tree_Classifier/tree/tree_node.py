import pandas as pd
class TreeNode:
    def __init__(self, x: pd.DataFrame, y: pd.DataFrame):
        self.left = None
        self.observations = x
        self.labels = y
        self.right = None
        self.column = None
    
    def insertNode(self,data):
        if self.left is None:
            self.left = Node(data)
        elif self.left is not None:
            self.insertNode(data)
        else:
            self.left = node
            return
        
        if self.right is not None:
            return self.insertNode(node)
        
        else:
            self.right = node
            return
        
    def printTree(self,node):
        if self.left is not None:
            self.printTree(self.left)
        if self.right is not None:
            self.printTree(self.right)
            
        print(node.value)

class Tree:
    def __init__(self,Node):
        self.root = Node

if __name__ == "__main__":
    trees = Node('root ')
    trees.insertNode(Tree.createNode('1'))
    trees.insertNode(Tree.createNode('2'))
    trees.insertNode(Tree.createNode('3'))
    trees.insertNode(Tree.createNode('4'))
    trees.insertNode(Tree.createNode('5'))
    trees.insertNode(Tree.createNode('6'))
    trees.insertNode(Tree.createNode('7'))
    
    
            