# BisecBiO
Matlab Code of "Near-Optimal Convex Simple Bilevel Optimization with a Bisection Method"

<h1 align="center"> Experiments Introduction for Bisec-BiO</h1>

# Bisec-BiO
This document introduces the MATLAB codes for implementing the bisection-based method (Bisec-BiO) for the simple bilevel optimization problem, which is presented in the article "Near-Optimal Convex Simple Bilevel Optimization with a Bisection Method".

All simulations are implemented using MATLAB R2023a on a PC running Windows 11 with an AMD (R) Ryzen (TM) R7-7840H CPU (3.80GHz) and 16GB RAM. To execute the code, it is necessary to add the appropriate pathways within the code before running it.

We consider the following three problems in the experiments motivated by the examples proposed in the appendix.

## Minimum Norm Solution Problem (MNP)
Before running the code, you should download the YearPredictionMSD dataset from https://archive.ics.uci.edu/dataset/203/yearpredictionmsd, rename it as "YearPredictionMSD.txt" and put it in the "MNP" folder. Then you can run the file "main_LeastSquare.m" in the "MNP" folder directly for the experiment.

## Logistic Regression Problem (LRP)
Before running the code, you should download the LIBSVM package from https://www.csie.ntu.edu.tw/~cjlin/libsvm/#download, generate "libsvmread.mexw64" file using the "make.m" file in the LIBSVM Matlab package and add the appropriate pathways. Moreover, you should download the dataset "a1a.t" from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t and rename it as "a1a.t" (We have prepared the required files and datasets mentioned above in our folder). Then you can run the file "main_Logistic_a1a.m" in the "LRP" folder directly for the experiment.

## Sparse Solution of Least Squares Regression Problem (SSP)
The dataset requirement is identical to the MNP problem, you can run the file "main_LeastSquare.m" in the "SSP" folder directly for the experiment.
