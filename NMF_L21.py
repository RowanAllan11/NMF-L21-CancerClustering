import numpy as np
import pandas as pd
from accuracy import cluster_acc
from NMF_max1 import gene_filter, cluster_assignment

# Model origin: https://github.com/alejandrods/Analysis-of-the-robustness-of-NMF-algorithms/blob/master/algorithm/Analysis_of_the_robustness_of_NMF_algorithms.ipynb

class L21normNMF():
    def __init__(self):
        pass

    def initialise_matrices(self):
        rng = np.random.RandomState(self.seed) # Random number generator
        W = rng.rand(self.g, self.k)
        H = rng.rand(self.k, self.n)
        return W, H

    def compute(self, A):
        W,H = self.initialise_matrices()
        tol = 1e-5

        for step in range(self.n_iter):
            D = np.diag(1 / np.sqrt(np.sum(np.square(A - W.dot(H)), axis=0)))

            Wu = W * (A.dot(D).dot(H.T))/(W.dot(H).dot(D).dot(H.T)+1e-10)
            Hu = H * (Wu.T.dot(A).dot(D))/(Wu.T.dot(Wu).dot(H).dot(D)+1e-10)

            e_W = np.sqrt(np.sum((Wu-W)**2, axis=(0,1)))/W.size
            e_H = np.sqrt(np.sum((Hu-H)**2, axis=(0,1)))/H.size

            if e_W<tol and e_H<tol:
                print("step is:", step)
                break

            W = Wu
            H = Hu

        return W, H    

    def fit(self, A, n_components, n_iter, seed):
        # Store shape and parameters
        self.g, self.n = A.shape  # g = genes, n = samples
        self.k = n_components
        self.n_iter = n_iter
        self.seed = seed

        W, H = self.compute(A)

        clusters = cluster_assignment(W, H)
        
        return W, H, clusters

def run_model(A,labels, n_components, n_iter, seed, n_runs):
    model = L21normNMF()
    accuracies = []

    for i in range(n_runs):
        W, H, clusters = model.fit(A, n_components = n_components, n_iter=n_iter, seed=seed + i)
        A1 = gene_filter(W, A)
        W1, H1, clusters2 = model.fit(A1, n_components=n_components, n_iter=n_iter, seed=seed + 1000 + i)
        acc = cluster_acc(labels, clusters2)
        accuracies.append(acc)

    mean_accuracy = np.mean(accuracies)
    standard_error = np.std(accuracies, ddof=1) / np.sqrt(len(accuracies))
    print(f"{mean_accuracy:.2f} ± {standard_error:.2f}")
    return mean_accuracy, standard_error

if __name__ == '__main__':
    labels = np.load("Datasets/GSE4913/carc(K=2).npy")
    A = np.load("Datasets/GSE4913/GSE4913.npy")
    #df = pd.read_csv("datasets/LK/LK.txt", header=0, sep="\t")
    #A = df.values
    run_model(A, labels, n_components=2, n_iter=500, seed=23, n_runs=100)

