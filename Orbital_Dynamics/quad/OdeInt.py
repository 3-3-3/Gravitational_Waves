import numpy as np

class OdeInt:
    def __init__(x_1, x_2, eps, h_0=None, h_min=0):
        s.MAXSTP = 50000
        s.eps = eps
        s.x_1 = x_1; s.x_2 = x_2

        if h_0 == None:
            h_0 = (s.x_1 - x.x_2) / 2
        else:
            s.h_0 = h_0

        s.h_min = h_min
        s.y = y; s.dydx = dydx
        s.stepper = stepper
        s.out = Output(k_max, save_every, n_var, x_1, x_2)
        s.nstep = 0
        #s.x; s.h

class Output:
    def __init__(s, k_max, save_every, n_var, x_1, x_2, dense=False):
        s.k_max = k_max
        s.save_every = save_every #intervals to save at
        s.saved = 0
        s.dense = dense
        s.n_var = n_var #number of variables to store for each x

        s.x_1 = x_1; s.x_2 = x_2

        x_save = np.empty(k_max)
        y_save = np.empty((n_var, k_max))

    def resize(s):
        s.k_max = 2 * k_max
        s.x_save.resize(k_max)
        s.y_save.resize(k_max)

    def save(s, x, y):
        s.count += 1
        if s.count == s.k_max:
            s.resize()

        for i in range(n_var):
            y_save[i][count] = y[i]

        x_save[count] = x

    def out(s):
        return (s.x_save[:count], s.y_save[:, :count])

class Adaptive_Euler_Stepper:
    def __init__(s, x_0, y_0, dxdy, tol, h_0):
        s.x = x_0
        s.y = y_0
        s.dxdy = dxdy
        s.tol = tol
        s.h_0 = h_0



    def euler_step(s, h, args=[]):
        y_new = s.y + s.dxdy(s.x_0, *args) * h
        x_new = s.x + h
        return (x_new, y_new)


    def refine(s, h):
        y_1 = euler_step(h)
        y_2 = euler_step(h / 2)

        if (y_2 - y_1) < tol:
            return y_2
        else:
            return refine(h / 2)
