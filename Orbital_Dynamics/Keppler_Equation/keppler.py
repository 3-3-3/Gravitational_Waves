import numpy as np
from scipy.optimize import fsolve

def newton(f, f_p, x_0, tol,f_args=[],f_p_args=[]): #Logical option for Keppler because we have good x_0 and f_p
    '''
    f: function
    f_p: function first derivative
    x_0: initial
    f_args: additional arguments for f
    f_p_args: additional arguments for f_p
    '''
    if np.abs(f(x_0, *f_args)) < tol:
        return x_0

    else:
        m = f_p(x_0, *f_p_args)
        x_new = (m * x_0 - f(x_0, *f_args)) / m
        return newton(f, f_p, x_new, tol,f_args=f_args, f_p_args=f_p_args)

def solve_keppler(t_array, T, ecc):
    '''
    t_array: times to evaluate at
    T: Orbital period
    ecc: Orbital eccentricity
    '''
    def keppler(psi, t, T, ecc):
        '''
        psi: function, to be solved for. In this case, we take it to be
        a variable for given T and t and will numerically solve for its root
        '''
        return psi - ecc * np.sin(psi) - (2 * np.pi / T) * t

    def d_keppler(psi, ecc):
        return 1 - ecc * np.cos(psi)

    #Initial guess; as most likely value of sin is zero
    #Simply use (2 * np.pi / T) * t for now

    return np.array([
        newton(keppler, d_keppler, (2 * np.pi / T) * t, 1e-5, f_args=[t, T, ecc], f_p_args=[ecc])
            for t in t_array])
