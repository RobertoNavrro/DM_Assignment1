from data.data_loader import load_credit_data, createMatrix, load_eclipse_data, load_eclipse_testdata
from tree.tree import tree_pred, tree_grow, tree_pred_b, tree_grow_b, tree_grow_limited
import numpy as np

def main():

    obs, labels = load_eclipse_data()
    testobs, testlabels = load_eclipse_testdata()
    
    #treeA = tree_grow(obs,labels, 2, 1, obs.shape[1])
    #treeB = tree_grow_b(obs,labels, 15, 5, obs.shape[1], 100)
    treeC = tree_grow_b(obs,labels, 15, 5, 6, 100)
    #treeD = tree_grow_limited(obs,labels, 15, 5, obs.shape[1], 2)

    #predicted_labels1 = tree_pred  (testobs,treeA)
    #predicted_labels2 = tree_pred_b(testobs,treeB)
    predicted_labels3 = tree_pred_b(testobs,treeC)
    #predicted_labels4 = tree_pred  (testobs,treeD)
        
    #To look into the splits
    #values, counts = np.unique(testlabels, return_counts=True)
    #print(values, counts)

    #Check the matrixes for tree ABCD 
    #print(createMatrix(predicted_labels1,testlabels))
    #print(createMatrix(predicted_labels2, testlabels))
    print(createMatrix(predicted_labels3,testlabels))
    #print(createMatrix(predicted_labels4,testlabels))

    #obs, labels = load_credit_data()
    #tree = tree_grow(obs,labels,2,1,obs.shape[1])
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
    
    #predicted_labels = tree_pred(obs,tree)
    #print(createMatrix(predicted_labels,labels))
    
if __name__ == "__main__":
    main()