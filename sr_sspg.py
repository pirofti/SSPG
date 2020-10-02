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

import numpy as np
from sspg import sspg

## STEPS
def stepsize_lin(k):
    return 5/k

def stepsize_quad(k):
    return 1/(k**2)


## SSPG-SR

def grad_f(x, i, mu, *args):
    sample, dictionary, Delta, alpha, lam = args

    In = np.eye(dictionary.shape[1])

    y = (np.outer(dictionary[i,:],dictionary[i,:]) + alpha*In)@x - dictionary[i,:].T*sample[i]

    return y

def grad_h(y, i, mu, *args):
    sample, dictionary, Delta, alpha, lam = args

    m = sample.shape[0]

    beta = (Delta[i,:]@y)/np.linalg.norm(Delta[i,:])**2
    if np.absolute(beta) > m*lam*mu:
        x = m*lam*np.sign(beta)*Delta[i,:].T
    else:
        x = (1/mu)*beta*Delta[i,:].T

    return x




def sr_sspg(sample, dictionary, Delta, alpha=0.001, lam=50,
            x0=None, stepup=stepsize_lin, eps=1e-6, xstar=None):

    if x0 == None:
        x0 = np.ones(dictionary.shape[1])

    F = lambda x, args: (
            1/(2*sample.shape[0])*np.linalg.norm(dictionary@x - sample)**2+
            lam*np.linalg.norm(Delta@x,1)+
            alpha/2*np.linalg.norm(x)**2
    )

    x_list, F_list, err_list = sspg(F, grad_f, grad_h, x0, sample.shape[0],
            stepup, eps, xstar,
            sample, dictionary, Delta, alpha, lam)

    return x_list[-1], x_list, F_list, err_list
