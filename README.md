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
- `.data/`: containing the original .csv files, along with stored data necessary for running tasks
- `.helper_functions/`:
  - `arma_helper_functions`
  - `discrete_helper_functions`
  - `pca_helper_functions`
  - `comparison_helper_functions`

## Lab 3: Network traffic
**IMPORTANT**: In case datasets are not included in the deliverable, scenario 43 `capture20110811.pcap.netflow.labeled` and 51 `capture20110818.pcap.netflow.labeled` must be downloaded from https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-43/ and https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-51/ respectively and then placed under `./data/`. Then you are ready to review. The modified dataset for scenario 51, `capture20110818.pcap.netflow.labeled.remastered`, for avoiding loading issues is generated and stored in the Flow data discretization task. Please run this notebook before proceeding to Botnet profiling, flow classification and Bonus tasks as they make use of this file.

- `Sampling_Task.ipynb`: Reservoir sampling
- `Sketching_Task.ipynb`: Count-Min sketching
- `Flow_Data_Discretization_Task.ipynb`
- `Botnet_Profiling_Task.ipynb`
- `Flow_Classification_Task.ipynb`: NaiveBayer and RandomForest tested on packet and host level
- `Bonus_Task.ipynb`: adversarial attacks on applied learning algorithms
- `.data/`:
  - `silhouette_scores.json`: silhouette scores extracted in the Flow data discretization task for detecting the two featues yielding the most separable clusters
- `./helper_functions/`: helper functions, along with detailed comments, used for the various tasks
  - `sampling.py`
  - `sketching.py`
  - `flow_data_discretization.py`
  - `botnet_profiling.py`
  - `classification.py`

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
  - switch to the installed kernel
