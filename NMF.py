import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from utils import cluster_acc
from utils import max_norm

"""
Base structure for algorithm 1 which includes an embedded filter and cluster assignment using re-weighting of the
H matrix. The algorithm involves two NMF steps using both L21 normalisation and max normalisation.
"""

class BaseNMF:
    def __init__(self, n_components, n_iter, seed):
        self.n_components = n_components
        self.n_iter = n_iter
        self.seed = seed
        self.W = None
        self.H = None

    def NMF_1(self, A):
        raise NotImplementedError("subclass must be implemented")
    
    # Embedded filter, removes 50% of genes.
    def gene_filter(self, A):
        u = self.W.max(axis=1) - self.W.min(axis=1)
        threshold = np.quantile(u, 0.5)
        I = np.where(u >= threshold)[0]
        A1 = A[I, :]
        return A1

    def NMF_2(self, A1):
        raise NotImplementedError("subclass must be implemented")
    
    def cluster_assignment(self):
        H = self.H * self.W.max(axis=0)[:, np.newaxis]
        cluster_assignment = np.argmax(H, axis=0)
        return cluster_assignment
    
    def fit(self, A):            
            self.W, self.H = self.NMF_1(A)
            A1 = self.gene_filter(A)
            self.W, self.H = self.NMF_2(A1)
            cluster_assignment = self.cluster_assignment()
            return cluster_assignment

# Scikit-learn NMF model paired with max norm function in "utils.py".
class NMF_Max(BaseNMF):
    def NMF_1(self, A):
        Initialise = NMF(n_components=self.n_components, init="random", solver="mu", 
                         beta_loss="kullback-leibler", max_iter=self.n_iter, random_state=self.seed)
        W = Initialise.fit_transform(A)
        H = Initialise.components_
        
        W = max_norm(W)
        return W, H

    def NMF_2(self, A1):
        Initialise = NMF(n_components=self.n_components, init="random", solver="mu", 
                         beta_loss="kullback-leibler", max_iter=self.n_iter, random_state=self.seed)
        W = Initialise.fit_transform(A1)
        H = Initialise.components_

        return W, H

# Custom L21 norm robust NMF model, code is adapted from alejandrods.
# https://github.com/alejandrods/Analysis-of-the-robustness-of-NMF-algorithms/blob/master/algorithm/Analysis_of_the_robustness_of_NMF_algorithms.ipynb

class NMF_L21(BaseNMF):
    def NMF_1(self, A):
        rng = np.random.RandomState(self.seed)
        g, n = A.shape
        k = self.n_components

        W = rng.rand(g, k)
        H = rng.rand(k, n)

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
    
    def NMF_2(self, A1):
        rng = np.random.RandomState(self.seed)
        g, n = A1.shape
        k = self.n_components

        W = rng.rand(g, k)
        H = rng.rand(k, n)

        tol = 1e-5

        for step in range(self.n_iter):
            D = np.diag(1 / np.sqrt(np.sum((A1 - W @ H) ** 2, axis=0) + 1e-10))

            Wu = W * (A1 @ D @ H.T) / (W @ H @ D @ H.T + 1e-10)
            Hu = H * (Wu.T @ A1 @ D) / (Wu.T @ Wu @ H @ D + 1e-10)

            e_W = np.linalg.norm(Wu - W) / W.size
            e_H = np.linalg.norm(Hu - H) / H.size

            if e_W < tol and e_H < tol:
                print("NMF_2 converged at step:", step)
                break

            W, H = Wu, Hu

        return W, H

# Function for multiple run accuracy loop of the algorithm with the specified model.    
def run_model(model_class, A, labels, n_components, n_iter, n_runs, seed):
    accuracies = []

    for i in range(n_runs):
        
        # Creates an instance of the specific class passed to the function.
        model = model_class(n_components=n_components, n_iter=n_iter, seed=seed+101+i)
        
        cluster_assignment = model.fit(A)

        acc = cluster_acc(labels, cluster_assignment)
        accuracies.append(acc)

    mean_accuracy = np.mean(accuracies)
    standard_error = np.std(accuracies, ddof=1) / np.sqrt(len(accuracies))
    
    print(f"Results for {model_class.__name__}:")
    print(f"Mean Accuracy: {mean_accuracy:.2f} ± {standard_error:.2f}")
    return mean_accuracy, standard_error

if __name__ == '__main__':
    
    # Load data and labels.
    labels = np.load("datasets/Medullo/Medullo(K=2).npy")
    df = pd.read_csv("datasets/Medullo/Medullo.txt", header=0, sep="\t")
    A = df.values   
    
    # Uniform noise.
    mu = 0.05
    lamda = mu * np.max(A)
    A_noisy = A + lamda * np.random.rand(*A.shape)

    """
    A - without noise.
    A_noisy - with noise.

    Run specified model with desired data, labels, number of componenets, iterations, runs and seed.
    """

    run_model(NMF_Max, A_noisy, labels, n_components=2, n_iter=500, n_runs=100, seed=50)
    run_model(NMF_L21, A_noisy, labels, n_components=2, n_iter=500, n_runs=100, seed=50)
