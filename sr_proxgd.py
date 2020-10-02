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

import cvxpy as cp
import numpy as np
from numpy import linalg

def sr_proxgd(sample, dictionary, Delta, alpha, lam, x0, eps, xstar):
    x_old = 0
    x_new = x0
    x_list, err_list = [x_new], []
    m = dictionary.shape[0]
    n = dictionary.shape[1]
    In = np.eye(n)
    E = np.real(linalg.eigvals((1/m)*dictionary.T@dictionary + alpha*In))
    L = E.max()
    eta = 1/L
    iter = 0

    while np.linalg.norm(x_new - xstar)**2 > eps:
        x_old = x_new
        iter = iter + 1

        # y = x_old - eta * ((dictionary.T@dictionary + alpha*In)@x_old
        #                   - dictionary.T @ sample)
        y = x_old - eta * ((1/m)*(dictionary.T@(dictionary@x_old - sample))
                           + alpha*x_old)

        z = cp.Variable(shape=n)
        cost = (L/2)*cp.sum_squares(z - y) + lam*cp.norm(Delta@z, 1)
        prob = cp.Problem(cp.Minimize(cost))
        prob.solve(solver=cp.CVXOPT, verbose=False)
        # print('Solver status: {}'.format(prob.status))

        # Check for error.
        if prob.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")

        x_new = z.value

        x_list.append(x_new)
        err_list.append(np.linalg.norm(x_new - xstar))

        # print("[", iter, "] Error", np.linalg.norm(x_new - xstar))

    return x_list[-1], x_list, err_list
