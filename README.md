# About:
In the process of replicating results from "Impact of the Choice of Normalization Method on Molecular Cancer Class Discovery Using Nonnegative Matrix Factorization" using python instead of MatLab. The NMF script thus far uses the maximum normalisation method with the embedded filter as this showed the best performance.
The core Leukemia and Medulloblastoma datasets have been tested for the "mean of clustering accuracies from a 100 runs of NMF together with the standard error of the mean" so far. However the results vary slightly from the original papers results and I am not sure why?

## Results Table:
| Datasets           | Max Norm (Yang et al.) | Max Norm (Python/Rowan) |
|--------------------|------------------------|--------------------------|
| Leukemia (k=2)     | 100 ± 0.00             | 0.98 ± 0.00              |
| Leukemia (k=3)     | 100 ± 0.00             | 0.98 ± 0.00              |
| CNS                | 96.88 ± 0.07           |                          |
| Medulloblastoma    | 61.76 ± 0.00           | 0.61 ± 0.00              |
