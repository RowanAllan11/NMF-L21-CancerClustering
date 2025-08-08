import pandas as pd
import numpy as np

#df2 = pd.read_csv("datasets/Medullo/labels.txt", sep="\t", header=None)
#txt = open("Blo3/blo3.txt")

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

def assign_labels(all_samples, recurrence_samples):
    """
    Assign labels to samples:
    1 if sample is in recurrence_samples,
    0 otherwise.

    Args:
        all_samples (list of str): List of all sample IDs.
        recurrence_samples (list of str): List of sample IDs with recurrence.

    Returns:
        list of int: Labels corresponding to all_samples.
    """
    recurrence_set = set(recurrence_samples)
    labels = [1 if sample in recurrence_set else 0 for sample in all_samples]
    df2 = pd.DataFrame(labels)
    column = df2[0]
    ints = pd.to_numeric(column, downcast="integer")
    labels = ints.to_numpy()
    print(labels.shape)
    np.save("Datasets/GSE4913/carc(K=2)", labels)
    return labels



all_samples = ['GSM110388', 'GSM110392', 'GSM110394', 'GSM110395', 'GSM110396', 'GSM110397', 'GSM110398', 'GSM110399', 'GSM110400', 'GSM110401', 'GSM110402', 'GSM110406', 'GSM110407', 'GSM110409', 'GSM110410', 'GSM110411', 'GSM110412', 'GSM110413', 'GSM110414', 'GSM110415', 'GSM110416', 'GSM110417', 'GSM110418', 'GSM110419', 'GSM110420', 'GSM110421', 'GSM110422', 'GSM110423', 'GSM110424', 'GSM110425', 'GSM110426', 'GSM110427', 'GSM110428', 'GSM110429', 'GSM110430', 'GSM110431', 'GSM110432', 'GSM110433', 'GSM110434', 'GSM110435', 'GSM110436', 'GSM110437', 'GSM110438', 'GSM110440', 'GSM110441', 'GSM110444', 'GSM110445', 'GSM110446', 'GSM110449', 'GSM110451']
recurrence_samples = [
    'GSM110388', 'GSM110392', 'GSM110394', 'GSM110402', 'GSM110411', 
    'GSM110412', 'GSM110417', 'GSM110422', 'GSM110426', 'GSM110429', 
    'GSM110433', 'GSM110436', 'GSM110440', 'GSM110441', 'GSM110444', 
    'GSM110445', 'GSM110446', 'GSM110449', 'GSM110451'
]




all = ['GSM877126', 'GSM877127', 'GSM877128', 'GSM877129', 'GSM877130', 'GSM877131',
        'GSM877132', 'GSM877133', 'GSM877134', 'GSM877135', 'GSM877136', 'GSM877137', 
        'GSM877138', 'GSM877139', 'GSM877140', 'GSM877141', 'GSM877142', 'GSM877143', 
        'GSM877144', 'GSM877145', 'GSM877146', 'GSM877147', 'GSM877148', 'GSM877149', 
        'GSM877150', 'GSM877151', 'GSM877152', 'GSM877153', 'GSM877154', 'GSM877155', 
        'GSM877156', 'GSM877157', 'GSM877158', 'GSM877159', 'GSM877160', 'GSM877161', 
        'GSM877162', 'GSM877163', 'GSM877164', 'GSM877165', 'GSM877166', 'GSM877167', 
        'GSM877168', 'GSM877169', 'GSM877170', 'GSM877171', 'GSM877173', 'GSM877174', 
        'GSM877175', 'GSM877176', 'GSM877177', 'GSM877178', 'GSM877179', 'GSM877180', 
        'GSM877181', 'GSM877182', 'GSM877183', 'GSM877184', 'GSM877185', 'GSM877186', 
        'GSM877187', 'GSM877188']

subtype = ['subtype: 2.1', 'subtype: 2.1', 'subtype: 2.1', 'subtype: 2.1', 'subtype: 1.1', 
           'subtype: 1.2', 'subtype: 1.3', 'subtype: 2.1', 'subtype: 2.2', 'subtype: 2.2', 
           'subtype: 2.2', 'subtype: 2.2', 'subtype: 2.1', 'subtype: 2.2', 'subtype: 2.1', 
           'subtype: 1.1', 'subtype: 1.1', 'subtype: 1.3', 'subtype: 1.2', 'subtype: 1.1', 
           'subtype: 1.3', 'subtype: 1.2', 'subtype: 1.3', 'subtype: 2.2', 'subtype: 2.2', 
           'subtype: 1.1', 'subtype: 1.3', 'subtype: 2.1', 'subtype: 2.2', 'subtype: 1.2', 
           'subtype: 2.1', 'subtype: 2.2', 'subtype: 1.1', 'subtype: 1.2', 'subtype: 2.2', 
           'subtype: 2.2', 'subtype: 1.2', 'subtype: 2.2', 'subtype: 2.1', 'subtype: 1.1', 
           'subtype: 2.2', 'subtype: 2.2', 'subtype: 1.3', 'subtype: 2.1', 'subtype: 1.2', 
           'subtype: 2.1', 'subtype: 1.1', 'subtype: 2.1', 'subtype: 2.2', 'subtype: 1.1', 
           'subtype: 2.2', 'subtype: 2.1', 'subtype: 1.1', 'subtype: 1.3', 'subtype: 1.1', 
           'subtype: 1.2', 'subtype: 2.1', 'subtype: 2.2', 'subtype: 1.1', 'subtype: 1.2', 
           'subtype: 2.2', 'subtype: 2.2']


#df = pd.read_csv("Datasets/GSE35896/GSE35896.txt", sep="\t", header=1)
#df = df.drop(columns=['1007_s_at'])
#df = df.dropna()
#np.save("Datasets/GSE35896/series", df)
#print(df.shape)


def assign_labels(all, subtype):
   d = {"accession": all, "subtype": subtype}
   df = pd.DataFrame(data=d)
   first = df["subtype"].str.startswith("subtype: 2")
   second = df['subtype'].str.startswith("subtype: 1")
   df.loc[first, "subtype"] = 1
   df.loc[second, "subtype"] = 0
   column = df.iloc[:, 1].tolist()
   labels = pd.to_numeric(column, downcast="integer")
   np.save("Datasets/GSE35896/adeno(K=2)", labels)
   return print(labels.shape)

samples_remove = ['ID_REF', 'GSM110391', 'GSM110439', 'GSM110442', 'GSM110443', 'GSM110447', 'GSM110448', 'GSM110450', 'GSM110452', 'GSM110453']
df = pd.read_csv("datasets/GSE4913/series.txt", header=0, sep="\t")
df = df.dropna()
df = df.drop(columns=samples_remove)
#df = df.drop(index=0)
df = df.reset_index(drop=True)
df = df + 10
df.columns = range(df.shape[1]) 
np.save("Datasets/GSE4913/GSE4913", df)
print(df.head())


