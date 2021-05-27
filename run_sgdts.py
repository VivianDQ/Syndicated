import numpy as np
import random
import math
import os
import argparse
import importlib
from algorithms.SGD_TS import *
from algorithms.AutoTuning import *
data_generator = importlib.import_module('algorithms.data_generator')

parser = argparse.ArgumentParser(description='simulations')
parser.add_argument('-data', '--data', type=str, default = 'simulations', help = 'can be movielens')
args = parser.parse_args()

T = 10000
rep = 10
J = [0, 0.01, 0.1, 1, 10] # tuning set for exploration para \alpha
etas = [0.01, 0.1, 1, 10] # tuning set for step size
algo = 'sgdts'
datatype = args.data

if datatype == 'simulations':
    sigma, d, K = 0.1, 10, 100
elif datatype == 'movielens':
    sigma, d, K = 0.5, 20, 1000
paras = {
    'eta0': etas,
    'alpha1': J,
    'alpha2': J,

}
print('tuning set of explore1, explore2, step size are {}'.format(paras))

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
reg_syndicated = np.zeros(T)
reg_tl = np.zeros(T)
reg_op = np.zeros(T)
reg_tl_combined = np.zeros(T)

methods = {
    'auto': '_syndicated',
    'op': '_op',
    'tl_combined': '_tl_combined',
}

for i in range(rep):
    np.random.seed(i+1)
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
    algo_class = SGD_TS(bandit, T)
    
    fcts = {
        k: getattr(algo_class, algo+methods[k]) 
        for k,v in methods.items()
    }
    
    reg_tl += fcts['auto']( tau, {'eta0': paras['eta0']} ) # tl
    reg_tl_combined += fcts['tl_combined'](tau, paras) # tl_combined
    reg_syndicated += fcts['auto'](tau, paras) # syndicated
    reg_op += fcts['op']( tau, {'eta0': paras['eta0']} ) # op
    
    print("op {}, tl {}, syndicated {}, combined {}".format(
        reg_op[-1], reg_tl[-1], reg_syndicated[-1], reg_tl_combined[-1]))
    
    result = {
        'tl': reg_tl/(i+1),
        'op': reg_op/(i+1),
        'syndicated': reg_syndicated/(i+1),
        'tl_combined': reg_tl_combined/(i+1),
    }
    for k,v in result.items():
        np.savetxt(path + k, v)   