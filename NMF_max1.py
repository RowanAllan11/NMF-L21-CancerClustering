import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from accuracy import cluster_acc

def SK_NMF(k, A):
    Initialise = NMF(n_components=k, init="random", solver="mu", beta_loss="kullback-leibler", max_iter=500)
    W = Initialise.fit_transform(A)
    H = Initialise.components_
    return W, H

def max_norm(W):
    W1 = W / W.max(axis=0)
    return W1

def gene_filter(W, A):
    # Embedded filter and maximum norm
    u = W.max(axis=1) - W.min(axis=1)
    threshold = np.quantile(u, 0.5)
    I = np.where(u >= threshold)[0]
    A1 = A[I, :]
    return A1

def cluster_assignment(W1, H1):
    H2 = H1 * W1.max(axis=0)[:, np.newaxis]
    cluster_assignments = np.argmax(H2, axis=0)
    return cluster_assignments

if __name__ == '__main__':
    # File path for gene expression data
    df = pd.read_csv("datasets/LK/LK.txt", header=0, sep="\t")
    labels = np.load("Datasets/LK/LK(K=2).npy")
    #Array = np.load("Datasets/GSE35896/series.npy")


    # Convert df to array
    A = df.values
    k = 2
    # Accuracy score list for each nmf run
    accuracies = []

    for i in range(100):
        W, H = SK_NMF(k, A)
        W_norm = max_norm(W)
        A1 = gene_filter(W_norm, A)
        W1, H1 = SK_NMF(k, A1)
        cluster_assignments = cluster_assignment(W1, H1)
        acc = cluster_acc(labels, cluster_assignments)
        accuracies.append(acc)

    mean_accuracy = np.mean(accuracies)
    standard_error = np.std(accuracies, ddof=1) / np.sqrt(len(accuracies))
    print(f"{mean_accuracy:.2f} ± {standard_error:.2f}")
