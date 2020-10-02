# Copyright (c) 2019-2020 Paul Irofti <paul@irofti.net>
# 
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
# 
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

from gensr_sspg import gensr_sspg, stepsize_lin
from gensr_proxgd import gensr_proxgd
from gensr_cvx import gensr_cvx
from reslimit import limit_memory

from sklearn.datasets import make_sparse_coded_signal
import numpy as np
import matplotlib as mpl
mpl.use("pgf")
from matplotlib import pyplot as plt
import pickle
import scipy


# SETUP
runtest = False     # run simulations or load results from disk?

n_nonzero_coefs = 3     # sparsity (s)
n_samples = 1           # number of signals (N)

alpha = 0.2    # l2 weight
lam = .0005     # l1 weight
eps = 1e-6      # precision
rounds = 10     # sspg rounds

datasize = 2.5 * 1024 * 1024 * 1024     # limit memory usage

prefix = 'gensr-errs'

cvx_errs, prox_errs, sspg_errs = [], [], []
atoms = range(30, 101, 20)
fname = '{0}-alpha{1}-lam{2}-eps{3}'.format(prefix,
        alpha, lam, eps)


# TEST

def save_test():
    with open('data/{0}-n{1}.dat'.format(fname, n_components), 'wb') as fp:
        pickle.dump(prox_errs, fp)
        pickle.dump(sspg_errs, fp)

def plot_test(i):
    plt.subplot(2, 2, i)

    t = [ i['time'] for i in prox_errs ]
    err = [ i['error'] for i in prox_errs ]
    plt.plot(t, err, 'r', label='prox', marker='o')

    for r in range(rounds):
        t = [ i['time'] for i in sspg_errs[r] ]
        err = [ i['error'] for i in sspg_errs[r] ]
        if r == 0:
            m_label='sspg'
        else:
            m_label=None
        plt.plot(t, err, 'g', label=m_label)

    ax = plt.gca()
    ax.set_yscale('log')
    if i % 2 == 0:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
    if i < 3:
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
    # else:
        # labels = ax.get_yticklabels()
        # labels[0] = ""
        # ax.set_yticklabels(labels)
    # ax.set_ylim([eps, None])

    ax.set_title('$m = {0}$'.format((5+(i-1)*25)*6), y=0.38)
    plt.xlabel('time (s)')
    plt.ylabel('$\|x - x^\star\|^2$')

    if i == 2:
        plt.legend()

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    if i == 4:
        plt.savefig('img/{0}-n{1}.pdf'.format(fname, n_components))
        plt.show()

    # plt.close(plt.gcf())

def load_test():
    with open ('data/{0}-n{1}.dat'.format(fname, n_components), 'rb') as fp:
        prox_errs = pickle.load(fp)
        sspg_errs = pickle.load(fp)
    return prox_errs, sspg_errs


i = 1
for n_components in atoms:
    limit_memory(datasize)

    if runtest:
        print("----------")
        print("atoms", n_components)
        n_features = n_components * 4

        # Generate sparse signals using a dictionary
        samples, dictionary, codes = make_sparse_coded_signal(
                n_samples=n_samples,
                n_components=n_components,
                n_features=n_features,
                n_nonzero_coefs=n_nonzero_coefs,
                random_state=0)
        Delta = np.random.standard_normal((n_features, n_components))
        # x0 = np.ones(dictionary.shape[1])
        x0 = np.random.standard_normal(dictionary.shape[1])*500

        # for sample in samples.T:
        sample = 500*samples    # XXX
        sample = sample - scipy.mean(sample)
        sample = sample / np.linalg.norm(sample)

        cvx_x = gensr_cvx(sample, dictionary, Delta, alpha, lam)

        prox_x, _, prox_errs = gensr_proxgd(sample, dictionary, Delta,
                alpha, lam, x0, eps, cvx_x)
        print("Prox iters:", len(prox_errs))

        sspg_x = []
        sspg_errs = []
        for r in range(rounds):
            x, _, _, errs = gensr_sspg(sample, dictionary, Delta,
                    alpha, lam, x0, stepsize_lin, eps, cvx_x)
            print("SSPG iters:", len(errs))
            sspg_x.append(x)
            sspg_errs.append(errs)

            # t = np.array([ e['time'] for e in sspg_errs ])
            # err = np.array([ e['error'] for e in sspg_errs ])
            # if r == 0:
            #     avgerr = err.copy()
            #     avgt = t.copy()
            # elif len(err) < len(avgerr):
            #     avgerr[:len(err)] += err
            #     avgt[:len(t)] += t
            # else:
            #     err[:len(avgerr)] += avgerr
            #     avgerr = err.copy()
            #     t[:len(avgt)] += avgt
            #     avgt = t.copy()
        # err = avgerr / rounds
        # t = avgt / rounds

        save_test()

    else:
        prox_errs, sspg_errs = load_test()

    plot_test(i)
    i = i + 1

