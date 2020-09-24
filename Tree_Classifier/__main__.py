from data.data_loader import load_credit_data
from tree.trees import tree_grow, printTree

def main():
    obs, labels = load_credit_data()
    tree = tree_grow(obs,labels,2,1,obs.shape[1])
    print("Here is the tree!:")
    printTree(tree,0)
    
    
if __name__ == "__main__":
    main()