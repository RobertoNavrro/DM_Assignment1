from data.data_loader import load_credit_data, createMatrix
from tree.tree import tree_pred, tree_grow, tree_pred_b

def main():
    obs, labels = load_credit_data()
    tree = tree_grow(obs,labels,2,1,obs.shape[1])
    # incase you want to test out the tree_pred_b
    
    
    # tree1 = tree_grow(obs,labels,2,1,obs.shape[1])
    # tree2 = tree_grow(obs,labels,2,1,obs.shape[1])
    # tree3 = tree_grow(obs,labels,2,1,obs.shape[1])
    # tree4 = tree_grow(obs,labels,2,1,obs.shape[1])
    # tree_list = []
    # tree_list.append(tree)
    # tree_list.append(tree1)
    # tree_list.append(tree2)
    # tree_list.append(tree3)
    # tree_list.append(tree4)
    
    
    predicted_labels = tree_pred_b(obs,tree_list)
    # print(createMatrix(predicted_labels,labels))
    
    
if __name__ == "__main__":
    main()