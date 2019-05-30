# CS4035
Cyber Data Analytics CS4035, 2018-2019
Authors:
  - Kostantinos Chronas, 4923162
  - Ioannis Lelekas, 4742559

## Lab 1: Fraud Detection
- `Lab1.ipynb`: pipeline for the fraud detection task of Lab 1,
- `Lab1-Bonus.ipynb`: pipeline for the bonus task of Lab 1,
- `HelperFunctions.py`: implementation of helper functions for conducting ROC analysis and plotting the aggregate confusion matrix

## Lab 2: Anomaly detection
- `Familiarization Task.ipynb`: Data importing and preprocessing; simple AR(1) model on `L_T1`
- `ARMA Task.ipynb`: ARMA models for sensors `L_T1`, `L_T2`, `L_T4`, `L_T7`, `F_PU1` and `F_PU2`
- `Discrete Task.ipynb`: discrete models (SAX via sliding window) for the same sensors
- `PCA Task.ipynb`: PCA method for outlier detection
- `data/`: containing the original .csv files, along with stored data necessary for running tasks
- `helper_functions/`:
  - `arma_helper_functions`
  - `discrete_helper_functions`
  - `pca_helper_functions`
  - `comparison_helper_functions`

## Lab 3: Network traffic

## For dear reviewers:
- The assignments are implemented in Python and offered in form of Jupyter notebooks. We advise installing anaconda; for anaconda installation please refer to:
  - https://docs.anaconda.com/anaconda/install/
- Once Anaconda is installed:
  - open terminal in the folder containing the contents of to-be-assessed assignment,
  - create an environment with the supplied .yml file by running:
  `conda env create -f cda.yml`
  - activate the environment: `conda activate cda`
  - type `jupyter notebook`
  - open the notebooks
