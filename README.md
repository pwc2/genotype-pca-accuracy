# genotype-pca-accuracy

This repository contains the scripts and notebooks used for data analysis in the manuscript titled "Assessing the estimation of population structure by genotype principal component analysis".

## UK Biobank (UKB)

  - Notebooks for variant/sample QC, the hdpca comparison, and creating plots are located in `genotype-pca-accuracy/UKB/notebooks/`.

  - Scripts for running the PCA/SM estimator on the odd/even chromosome split, as well as on the full, unsplit data are located in `genotype-pca-accuracy/UKB/scripts/`.

The notebook `ukb, hgdp_1kg, 1kg - load scores, eigenvalues, and spectral moments.ipynb` contains functions used to load the results from the PCA/SM estimator (for UKB, 1KG+HGDP, and 1KG).

