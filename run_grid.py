import numpy as np
import random
import math
import os
import argparse
import importlib
from scipy.stats import truncnorm

from algorithms.LinUCB import *
from algorithms.LinTS import *
from algorithms.UCB_GLM import *
from algorithms.AutoTuning import *
data_generator = importlib.import_module('algorithms.data_generator')

parser = argparse.ArgumentParser(description='simulations')
parser.add_argument('-algo', '--algo', type=str, default = 'linucb', help = 'can also be lints, do not support glmucb')
parser.add_argument('-dist', '--dist', type=str, default = 'uniform', help = 'can also be uniform_fixed')
args = parser.parse_args()


T = 10**4
d = 5
rep = 5
K = 100
lamda = 1
delta = 0.1
sigma = 0.5
ub = 1/math.sqrt(d)
lb = -1/math.sqrt(d)
J = np.arange(0, 10.1, 0.5)
final = {k:[] for k in J}
dist = args.dist
algo = args.algo

for explore in J:
    reg_theory = np.zeros(rep)
    for i in range(rep):
        np.random.seed(i+1)
        theta = np.random.uniform(lb, ub, d)
        if dist == 'uniform':
            fv = np.random.uniform(lb, ub, (T, K, d))
        elif dist == 'uniform_fixed':
            fv = [np.random.uniform(lb, ub, (K, d))]*T
        context = data_generator.context
        bandit = context(K, T, d, sigma, true_theta = theta, fv=fv)
        bandit.build_bandit()

        if algo == 'linucb':
            algo_class = LinUCB(bandit, T)
        elif algo == 'lints':
            algo_class = LinTS(bandit, T)

        fcts = getattr(algo_class, algo + '_theoretical_explore')
        reg_theory[i] = fcts(lamda, delta, explore)[-1]
    final[explore] = [np.mean(reg_theory), np.std(reg_theory)]
    print("explore = {} done! avg cum reg {}, std of cum reg {}".format(explore, np.mean(reg_theory), np.std(reg_theory)))

theory = np.zeros(rep)
for i in range(rep):
    np.random.seed(i+1)
    theta = np.random.uniform(lb, ub, d)
    if dist == 'uniform':
        fv = np.random.uniform(lb, ub, (T, K, d))
    elif dist == 'uniform_fixed':
        fv = [np.random.uniform(lb, ub, (K, d))]*T
    context = data_generator.context
    bandit = context(K, T, d, sigma, true_theta = theta, fv=fv)
    bandit.build_bandit()

    if algo == 'linucb':
        algo_class = LinUCB(bandit, T)
    elif algo == 'lints':
        algo_class = LinTS(bandit, T)

    fcts = getattr(algo_class, algo+'_theoretical_explore')
    theory[i] = fcts(lamda, delta, -1)[-1]
    
final['theory'] = [np.mean(theory), np.std(theory)]    
print("theoretical explore done! avg cum reg {}, std of cum reg {}".format(np.mean(theory), np.std(theory)))