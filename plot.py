import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
from matplotlib import pylab

def draw_figure():
    plot_style = {
            'theory': ['-.', 'orange', '$\\bf{Theoretical-Explore}$'],
            'tl': ['-', 'black', '$\\bf{TL}$'],
            'op': [':', 'blue', '$\\bf{OP}$'],
            'syndicated': ['-', 'red', '$\\bf{Syndicated}$'],
            'tl_combined': ['--', 'green', '$\\bf{TL-Combined}$'],
        }
    plot_prior = {
            'syndicated': 1,
            'tl': 2,
            'tl_combined': 3,
            'theory': 4,
            'op': 5,
        }
    root = 'results/'
    if not os.path.exists('plots/'):
        os.mkdir('plots/')   
    cat = os.listdir(root)
    paths = []
    for c in cat:
        if 'movielens' not in c and 'simulation' not in c: continue
        folders = os.listdir(root+c)
        for folder in folders:
            paths.append(root + c + '/' + folder + '/')
    for path in paths:
        if 'grid' in path: continue
        algo = path.split('/')[-2]
        fn = path.split('/')[-3]
        if 'simulation' in fn:
            title = '$\\bf{Simulation\ for\ '
        elif 'movielens' in fn:
            title = '$\\bf{Movielens\ for\ '  
        if algo == 'linucb': prefix = 'LinUCB}$'
        elif algo == 'lints': prefix = 'LinTS}$'
        elif algo == 'glmucb': prefix = 'UCB-GLM}$'
        elif algo == 'sgdts': prefix = 'SGD-TS}$'
        title += prefix
        fig = plot.figure(figsize=(6,4))
        matplotlib.rc('font',family='serif')
        params = {'font.size': 18, 'axes.labelsize': 18, 'font.size': 12, 'legend.fontsize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12, 'axes.formatter.limits':(-8,8)}
        pylab.rcParams.update(params)
        leg = []
        keys = os.listdir(path)
        keys = sorted(keys, key=lambda kv: plot_prior[kv])
        y_label = '$\\bf{Cumulative\ Regret}$'
        for key in keys:
            if key not in plot_style.keys(): continue
            leg += [plot_style[key][-1]]
            data = np.loadtxt(path+key)
            T = len(data)
            plot.plot((list(range(T))), data, linestyle = plot_style[key][0], color = plot_style[key][1], linewidth = 2)
        loca = 'upper left' if algo == 'glmucb' and 'movielens' in fn else 'best'
        plot.legend((leg), loc=loca, fontsize=12, frameon=False)
        plot.xlabel('$\\bf{Iterations}$')
        plot.ylabel(y_label)
        plot.title(title)
        fig.savefig('plots/{}_{}.pdf'.format(algo, fn), dpi=300, bbox_inches = "tight")
        print('file in path {} plotted and saved as {}_{}.pdf'.format(path, algo, fn))

draw_figure()