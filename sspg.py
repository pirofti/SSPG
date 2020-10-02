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
from timeit import default_timer as timer

def sspg(F, grad_f, grad_h, x0, m, stepup, eps, xstar, *args):
    nochange = 0        # consecutive iterations no change counter
    x_old = 0           # previous iteration solution
    x_new = x0          # current iteraiton solution
    mu = 1              # regularization parameter
    iter = 0            # iteration counter

    # solutions, errors and function points throughout the iterations
    x_list, err_list = [x_new], []
    if F is None:
        F_list = []
    else:
        F_list = [F(x_new, args)]

    # stopping rule:
    #   if xstar is provided, stop when we are eps away from it
    #   if xstar is unknown, stop when no change happens between iterations
    iterchange = False
    if xstar is None:
        iterchange = True       # when tracking iteration changes,
        xstar = np.Infinity     # xstar becomes the previous (old) solution

    start = timer()
    err = np.linalg.norm(x_new - xstar)**2
    err_list.append({"iter" : iter, "error" : err, "time" : 0.0})
    while nochange < 20:
        iter = iter + 1
        mu = stepup(iter)
        i = np.random.randint(m)
        x_old = x_new

        y = x_old - mu * grad_f(x_old, i, mu, *args)
        x_new = y - mu * grad_h(y, i, mu, *args)
        if iterchange:
            xstar = x_old

        # check error
        err = np.linalg.norm(x_new - xstar)**2
        if err < eps:
            if not iterchange:       # stop if we reached the optimum solution
                break
            nochange = nochange + 1  # nochange between old and new solutions
        else:
            nochange = 0             # need to be consecutive nochange iters

        # report
        titer = timer()
        x_list.append(x_new)
        err_list.append({"iter" : iter, "error" : err, "time" : titer - start})
        if F is not None:
            F_list.append(F(x_new, args))

        # print("[", iter, "] Error", err, mu)

    # print("Local minimum occurs at:", x_new)
    # print("Number of steps:", len(x_list))

    return x_list, F_list, err_list
