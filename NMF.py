import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from accuracy import cluster_acc
from NMF_max1 import max_norm

# Base structure for algorithm 1 which includes an embedded filter with a default T=0.5 and cluster assignment using re-weighting of the
# H matrix. The algorithm involves two NMF steps using the desired normalisation technique.

class BaseNMF:
    def __init__(self, n_components, n_iter, seed):
        self.n_components = n_components
        self.n_iter = n_iter
        self.seed = seed
        self.W = None
        self.H = None

    def NMF_1(self, A):
        raise NotImplementedError("subclass must be implemented")
    
    def gene_filter(self, A):
        # Embedded filter and maximum norm
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

#Function to run a 100-iteration accuracy loop of the algorithm with the specified model.    
def run_model(model_class, A, labels, n_components, n_iter, n_runs, seed):
    accuracies = []

    for i in range(n_runs):
        # Create an instance of the specific class passed to the function
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
    # Load your data and labels here
    labels = np.load("Datasets/LK/LK(K=2).npy")
    df = pd.read_csv("datasets/LK/LK.txt", header=0, sep="\t")
    A = df.values   
    
    #Run the NMF_Max (scikit-learn) model
    run_model(NMF_Max, A, labels, n_components=2, n_iter=500, n_runs=100, seed=50)
