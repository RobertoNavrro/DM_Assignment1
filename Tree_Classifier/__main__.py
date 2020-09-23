from data.data_loader import load_credit_data
from tree.trees import tree_grow

def main():
    
    obs, labels = load_credit_data()
    tree = tree_grow(obs,labels,2,1,obs.shape[1])
    print(tree)
    for tup in tree:
        if tup is not None:
            tup.printSplit()

if __name__ == "__main__":
    main()