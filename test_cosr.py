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

from cosr_sspg import cosr_sspg, stepsize_lin
# from sr_proxgd import sr_proxgd
from cosr_cvx import cosr_cvx

from sklearn.datasets import make_sparse_coded_signal

import numpy as np
from matplotlib import pyplot as plt

from timeit import default_timer as timer

import pickle

from reslimit import limit_memory


# SETUP
n_nonzero_coefs = 3    # sparsity (s)
n_samples = 1          # number of signals (N)

alpha = 0       # l2 weight
lam = 5         # l1 weight
mu = 0.05       # learning rate
eps = 1e-3      # precision

datasize = 2.5 * 1024 * 1024 * 1024     # limit memory usage

prefix = 'cosr-atoms'


# TEST
limit_memory(datasize)

cvx_time, prox_time, sspg_time = [], [], []
# atoms = range(300, 301, 1)
atoms = range(80, 500, 20)
fname = '{0}-{1}-{2}-{3}'.format(prefix, atoms.start, atoms.stop, atoms.step)
for n_components in atoms:
    print("----------")
    print("atoms", n_components)
    n_features = round(n_components / 20)
    # Generate sparse signals using a dictionary
    sample, dictionary, codes = make_sparse_coded_signal(
        n_samples=n_samples,
        n_components=n_components,
        n_features=n_features,
        n_nonzero_coefs=n_nonzero_coefs,
        random_state=0)
    dictionary = np.eye(n_features)
    Delta = np.random.standard_normal((n_components, n_features))

    start = timer()
    cvx_x = cosr_cvx(sample, Delta, lam)
    end = timer()
    cvx_time.append(end-start)
    print("CVX time: ", end - start)
    # print("CVX: Sparsity", np.count_nonzero(Delta@cvx_x),
    #         "solution", Delta@cvx_x)

    # x0 = np.ones(dictionary.shape[1])
    # start = timer()
    # prox_x, _, prox_errs = sr_proxgd(sample, dictionary, Delta, alpha, lam,
    #                                 x0, eps, cvx_x)
    # end = timer()
    # prox_time.append(end-start)
    # print("Prox time: ", end - start)
    # print("Prox: Sparsity", np.count_nonzero(Delta@prox_x),
    #      "solution", Delta@prox_x)

    start = timer()
    sspg_x, _, _, sspg_errs = cosr_sspg(sample, dictionary, Delta, alpha, lam,
                                        None, stepsize_lin, eps, cvx_x)
    end = timer()
    sspg_time.append(end-start)
    print("SSPG time: ", end - start)
    # print("SSPG: Sparsity", np.count_nonzero(Delta@sspg_x),
    #       "solution", Delta@sspg_x)

# RESULTS
with open('{0}.dat'.format(fname), 'wb') as fp:
    pickle.dump(cvx_time, fp)
    pickle.dump(prox_time, fp)
    pickle.dump(sspg_time, fp)

# PLOTS
plt.plot(atoms, cvx_time, 'b', label='cvx')
plt.plot(atoms, prox_time, 'r', label='prox')
plt.plot(atoms, sspg_time, 'g', label='sspg')
plt.xlabel('atoms')
plt.ylabel('time (s)')
plt.legend()
plt.savefig('{0}.pdf'.format(fname))
plt.show()
