from data.data_loader import load_credit_data, createMatrix
from tree.trees import tree_grow, printTree, tree_pred

def main():
    obs, labels = load_credit_data()
    tree = tree_grow(obs,labels,20,5,obs.shape[1])
    # printTree(tree)
    predicted_labels = tree_pred(obs,tree)
    print(createMatrix(predicted_labels,labels))
    # print("Here is the tree!:")
    # printTree(tree)
    
    
if __name__ == "__main__":
    main()