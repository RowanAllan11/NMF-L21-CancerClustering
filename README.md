# About:
Molecular cancer class discovery is of great importance for tumour samples with unknown cancer types or subtypes, as their discovery can correspond to differences in prognosis, treatment response and disease mechanisms. Class discovery aims to uncover hidden structure in high-dimensional data that may reflect biological variation.

Non-negative matrix factorisation (NMF) has been shown to be a particularly effective technique for the task of clustering cancer samples in gene expression data. NMF decomposes the non-negative gene expression matrix into two lower dimensional matrices that capture underlying "metagenes" and their patterns across samples.

One major challenge in NMF is that the decomposition is not unique across different runs allowing different clustering results to be obtained. However, various normalisation techniques can be used which help alleviate this problem. Max normalisation makes NMF less sensitive to the initial selection of genes and more robust to data variation. L21 normalisation which has been proposed here as an alternative normalisaition choice, calculates the sum of the L2 norms of the columns. It does not square the error for each data point making it more resilient to outliers and noise than the standard NMF.

This study involved replicating results from the landmark paper; "Impact of the Choice of Normalization Method on Molecular Cancer Class Discovery Using Nonnegative Matrix Factorization" using python instead of MatLab. Their paper concluded the use of maximum normalisation with their unique embedded filter yielded the most accurate results. This study successfully recreated this technique as well as tested an alternative normalisation choice with and without added noise.

- NMF.py - Contains code to run both models.
- Utils.py - References the accuracy metric being used (Hungarian algorithm) and included the max norm function.

## Results Table:
| Dataset              | Max Norm | L21 Norm | Max Norm (with noise) | L21 Norm (with noise) |
|----------------------|----------|----------|------------------------|------------------------|
| Leukemia (k=2)       | 0.98 ± 0.00 | 0.86 ± 0.02 | 0.91 ± 0.00 | 0.95 ± 0.00 |
| Leukemia (k=3)       | 0.98 ± 0.00 | 0.94 ± 0.00 | 0.89 ± 0.01 | 0.95 ± 0.00 |
| CNS                  | 0.91 ± 0.01 | 0.80 ± 0.00 | 0.77 ± 0.01 | 0.81 ± 0.01 |
| Medulloblastoma      | 0.61 ± 0.00 | 0.62 ± 0.00 | 0.61 ± 0.00 | 0.62 ± 0.00 |
