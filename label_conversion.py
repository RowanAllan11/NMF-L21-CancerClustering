import pandas as pd
import numpy as np

df2 = pd.read_csv("datasets/Medullo/labels.txt", sep="\t", header=None)

# So after NMF of LK each sample is assigned a cluster (e.g. 0, 1, 2). Its in a numpy array and I want to use the accuracy score function which basically
# overlays them to see if they match. My labels list has either AML or ALL at the start of each row, need to convert either the ALL to the number (0, 1, 2) or
# assign a number to ALL/AML. Then I can overlay them and get the accuracy.

def labels_k3(df2):
    # Conversion to array, replacing labels with intergers
    column = df2[0]
    df2.loc[column.str.startswith('ALL') & column.str.endswith('T-cell'), 0] = "2"
    df2.loc[column.str.startswith('ALL') & column.str.endswith('B-cell'), 0] = "1"
    df2.loc[column.str.startswith('AML'), 0] = "0"
    column = pd.to_numeric(df2[0], downcast="integer")
    labels = df2[0].to_numpy()
    return np.save("Datasets/LK/LK(K=3)", labels)

def labels_k2(df2):
    # Conversion to array, replacing labels with intergers
    column = df2[0]
    df2.loc[column.str.startswith('ALL'), 0] = "1"
    df2.loc[column.str.startswith('AML'), 0] = "0"
    column = pd.to_numeric(df2[0], downcast="integer")
    labels = df2[0].to_numpy()
    return np.save("Datasets/LK/LK(K=2)", labels)

def labels_medullo(df2):
    column = df2[0]
    column.iloc[:25] = 0
    column.iloc[25:34] = 1
    ints = pd.to_numeric(column, downcast="integer")
    labels = ints.to_numpy()
    return np.save("Datasets/Medullo/Medullo(K=2)", labels)

labels_medullo(df2)
