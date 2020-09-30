from data.data_loader import load_credit_data, createMatrix
from tree.trees import tree_grow, tree_pred

def main():
    obs, labels = load_credit_data()
    tree = tree_grow(obs,labels,20,5,obs.shape[1])
    predicted_labels = tree_pred(obs,tree)
    print(createMatrix(predicted_labels,labels))
    
    
if __name__ == "__main__":
    main()