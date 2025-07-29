import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import accuracy_score

# File path for gene expression data
df = pd.read_csv("datasets/LK.txt", sep="\t", header=None)

# 5000 rows x 38 columns (GenesxSamples)
# Convert df to array
Array = df.values
print(Array.shape)

# Post-processing Method
# Inital NMF run using KL divergence
Initial = NMF(n_components=2, init="random", solver="mu", beta_loss="kullback-leibler", max_iter=500, random_state=42)

# Fit the model to data
W0 = Initial.fit_transform(Array)
H0 = Initial.components_ 

print("W0 shape:", W0.shape)
print("H0 shape:", H0.shape)

# Filter application and normalisation
# Normalise W0 - divide by max value
W = W0 / W0.max(axis=0)    # .max and axis 0 works down each column and returns a list with the max value for each e.g. [9, 8]

# Creating vector u which contains the difference between max and min values for each row (gene)
u = W.max(axis=1) - W.min(axis=1)

# Threshold to remove uninformative genes
threshold = np.quantile(u, 0.5)

# Keep genes I
I = np.where(u >= threshold)[0]     # [0] extracts the actual array of indices

# Filter to remove the uninformative genes 
A1 = Array[I, :]

print(A1.shape)

# Second NMF is run on A1, yielding W1 and W2
Initial = NMF(n_components=2, init="random", solver="mu", beta_loss="kullback-leibler", max_iter=500, random_state=43)
W1 = Initial.fit_transform(A1)
H1 = Initial.components_ 

# Obtain H2 by multiplying each row of H1 by the max value of corresponding W1 column
H2 = H1 * W1.max(axis=0)[:, np.newaxis]

# For each sample pick the cluster with the max score
cluster_assignments = np.argmax(H2, axis=0)

print(cluster_assignments)

# Loads truth Labels
df2 = pd.read_csv("datasets/LK3.txt", sep="\t", header=None)

# Conversion to array
df2.loc[df2[0].str.startswith('ALL'), 0] = "1"
df2.loc[df2[0].str.startswith('AML'), 0] = "0"
print(df2)
df2[0] = pd.to_numeric(df2[0], downcast="integer")
labels = df2[0].to_numpy()

# Accuracy score of model
print(accuracy_score(cluster_assignments, labels))



