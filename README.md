# Syndicated Bandits: A Framework for Auto Tuning Hyper-parameters in Contextual Bandit Algorithms

This repository is the official implementation of [Syndicated Bandits: A Framework for Auto Tuning Hyper-parameters in Contextual Bandit Algorithms].


## Requirements

To run the code, you will need 

```
Python3, NumPy, Matplotlib, R
```

For experiments on the real dataset Movielens 100K, the raw data should be put in an appropriate path. See details in Section ``Real datasets`` below for instructions.


## Commands

### Table 1

To get the results in Table 1 in the paper, run the following command:

```
python3 run.py -algo {algorithm} -dist {distribution}
```

In the above command, replace ``{algorithm}`` with ``linucb`` or ``lints`` to get the results for different algorithms. Replace ``{distribution}`` with ``uniform`` or ``uniform_fixed`` to run the experiments for the two scenarios when mentioned in the paper. When ``dist`` is set as ``uniform``, it means that the feature vectors are changing and re-simulated from the uniform distribution each round. When ``dist`` is set as ``uniform_fixed``, it means that the feature vectors are fixed for all rounds


### Figure 2

To get the results in Figure 2 in the paper, run the following command:

```
python3 run.py -algo {algorithm} -data {dataname}
```

In the above command, replace ``{algorithm}`` with ``linucb`` or ``lints`` or ``glmucb`` to compare the tuning methods in each algorithm. Replace ``{dataname}`` with ``simulations`` or ``movielens`` to run the experiments on different datasets.



### Figure 3 in Appendix

To get the results in Figure 2 in the appendix of the paper, run the following command:
```
python3 run_sgdts.py -data simulations
python3 run_sgdts.py -data movielens
```

## Real datasets

Get the following raw data from movielens official website and put it in the ``raw_data`` folder inside the ``data`` folder.

- Movieslens 100k dataset: you need ``u.data`` file for this dataset.

Since we run matrix factorization on the raw data, we also need the matrix factorization package `libpmf-1.41`. Get the package from its [official website](https://www.cs.utexas.edu/~rofuyu/libpmf/) and unzip the package and save the whole `libpmf-1.41` folder inside the ``data`` folder. You may need to read the instructions of that package to compile the program.

If it is your first time to run the experiments on ``movielens`` dataset, then our code will automatically preprocess the raw data, and it may take a while. Note that if the raw data does not exist or is not in the correct path, our code will report error.



## Plots

Numerical results will be saved in the ``results`` folder which is automatically created by our code. To produce the same plots as in our paper, run the following command, it will create a ``plots`` folder and all the figures will be saved there.

```
python3 plot.py
```
