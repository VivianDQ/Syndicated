import numpy as np
import random
import math
import os
import argparse
import importlib

from algorithms.LinUCB import *
from algorithms.LinTS import *
from algorithms.UCB_GLM import *
from algorithms.AutoTuning import *
data_generator = importlib.import_module('algorithms.data_generator')

parser = argparse.ArgumentParser(description='simulations')
parser.add_argument('-algo', '--algo', type=str, default = 'linucb', help = 'can also be lints, glmucb')
parser.add_argument('-data', '--data', type=str, default = 'simulations', help = 'can be movielens')
args = parser.parse_args()
    
T = 10000
rep = 10
lamda = 1
delta = 0.1
lamdas = [0.01, 0.1, 1] # tuning set for lambda
J = [0, 0.01, 0.1, 1, 10] # tuning set for exploration para \alpha
algo = args.algo
datatype = args.data
if datatype == 'simulations':
    d, K = 10, 100
    if 'lin' in algo:
        sigma = 0.1
    elif 'glm' in algo:
        sigma = 0.5
elif datatype == 'movielens':
    d, K = 20, 1000
    if 'lin' in algo:
        sigma = 1
    elif 'glm' in algo:
        sigma = 0.5
print('tuning set of explores {}, lamdas{}'.format(J, lamdas))

if not os.path.exists('results/'):
    os.mkdir('results/')
if not os.path.exists('results/' + datatype + '/'):
    os.mkdir('results/' + datatype + '/')
if not os.path.exists('results/' + datatype + '/' + algo + '/'):
    os.mkdir('results/' + datatype + '/' + algo + '/')
path = 'results/' + datatype + '/' + algo + '/'

if datatype == 'movielens':
    # check real data files exist:
    if not os.path.isfile('data/{}_users_matrix_d{}'.format(datatype, d)) or not os.path.isfile('data/{}_movies_matrix_d{}'.format(datatype, d)):
        print("{holder} data does not exist, will run preprocessing for {holder} data now. If you are running experiments for netflix data, then preprocessing might take a long time".format(holder=datatype))
        from data.preprocess_data import *
        process = eval("process_{}_data".format(datatype))
        process(d)
        print("real data processing done")   
    users = np.loadtxt("data/{}_users_matrix_d{}".format(datatype, d))
    fv = np.loadtxt("data/{}_movies_matrix_d{}".format(datatype, d))
    np.random.seed(0)
    thetas = np.zeros((rep, d))
    print(users.shape, fv.shape)
    for i in range(rep):
        thetas[i,:] = np.mean(users[np.random.choice(len(users), 100, replace = False), :], axis=0)

ub = 1/math.sqrt(d)
lb = -1/math.sqrt(d)
reg_theory = np.zeros(T)
reg_tl = np.zeros(T)
reg_op = np.zeros(T)
reg_syndicated = np.zeros(T)
reg_tl_combined = np.zeros(T)

methods = {
    'theory': '_theoretical_explore',
    'tl': '_tl',
    'op': '_op',
    'syndicated': '_syndicated',
    'tl_combined': '_tl_combined',
}

for i in range(rep):
    np.random.seed(i+1)
    if 'lin' in algo:
        if datatype == 'simulations':
            theta = np.random.uniform(lb, ub, d)
            fv = np.random.uniform(lb, ub, (T, K, d))
            context = data_generator.context
        elif datatype == 'movielens':
            theta = thetas[i, :]
            context = data_generator.movie
        bandit = context(K, T, d, sigma, true_theta = theta, fv=fv)
    elif 'glm' in algo:
        if datatype == 'simulations':
            theta = np.random.uniform(lb, ub, d)
            fv = np.random.uniform(-1, 1, (T, K, d))
            context_logistic = data_generator.context_logistic
            bandit = context_logistic(K, -1, 1, T, d, sigma, true_theta = theta, fv=fv)
        elif datatype == 'movielens':
            context_logistic = data_generator.movie_logistic
            theta = thetas[i, :]
            bandit = context_logistic(K, T, d, sigma, true_theta = theta, fv=fv)
    bandit.build_bandit()
    
    print(i, ": ", end = " ")
    if algo == 'linucb':
        algo_class = LinUCB(bandit, T)
    elif algo == 'lints':
        algo_class = LinTS(bandit, T)
    elif algo == 'glmucb':
        algo_class = UCB_GLM(bandit, T)
    
    fcts = {
        k: getattr(algo_class, algo+methods[k]) 
        for k,v in methods.items()
    }
    reg_theory += fcts['theory'](lamda, delta)
    reg_op += fcts['op'](J, lamda)
    reg_tl += fcts['tl'](J, lamda)
    reg_syndicated += fcts['syndicated'](J, lamdas)
    reg_tl_combined += fcts['tl_combined'](J, lamdas)
    
    print("theory {}, tl {}, op {}, syndicated {}, tl_combined {}".format(
        reg_theory[-1], reg_tl[-1], reg_op[-1], reg_syndicated[-1], reg_tl_combined[-1]))
    
    result = {
        'theory': reg_theory/(i+1),
        'tl': reg_tl/(i+1),
        'op': reg_op/(i+1),
        'syndicated': reg_syndicated/(i+1),
        'tl_combined': reg_tl_combined/(i+1),
    }
    for k,v in result.items():
        np.savetxt(path + k, v)   