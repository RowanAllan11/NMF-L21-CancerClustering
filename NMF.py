import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment

# File path for gene expression data
df = pd.read_csv("Datasets/Medullo/Medullo.txt", sep="\t", header=None)
labels = np.load("Datasets/Medullo/Medullo(K=2).npy")

# Convert df to array
Array = df.values
k = 2

def nmf_filter(Array, k):
    Initial = NMF(n_components=k, init="random", solver="mu", beta_loss="kullback-leibler", max_iter=500)
    W0 = Initial.fit_transform(Array)

    # Embedded filter and maximum norm
    W = W0 / W0.max(axis=0)
    u = W.max(axis=1) - W.min(axis=1)
    threshold = np.quantile(u, 0.5)
    I = np.where(u >= threshold)[0]
    A1 = Array[I, :]

    Initial2 = NMF(n_components=k, init="random", solver="mu", beta_loss="kullback-leibler", max_iter=500)
    W1 = Initial2.fit_transform(A1)
    H1 = Initial2.components_

    H2 = H1 * W1.max(axis=0)[:, np.newaxis]
    cluster_assignments = np.argmax(H2, axis=0)
    return cluster_assignments

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size

# Accuracy score list for each nmf run
accuracies = []

for i in range(100):
    cluster_assignments = nmf_filter(Array, k)
    acc = cluster_acc(labels, cluster_assignments)
    accuracies.append(acc)

mean_accuracy = np.mean(accuracies)
standard_error = np.std(accuracies, ddof=1) / np.sqrt(len(accuracies))
print(f"{mean_accuracy:.2f} ± {standard_error:.2f}")
