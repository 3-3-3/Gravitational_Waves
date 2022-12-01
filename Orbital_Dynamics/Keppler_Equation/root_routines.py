import numpy as np

def bisection(f, a, b, tol):
    '''
    f: function to find the root of
    a: Lower bracket
    b: Upper bracket
    tol: required tolerance
    '''
    bis = (b + a) / 2

    if np.abs(a - b) < tol:
        return bis

    else:
        if np.sign(f(bis)) == np.sign(f(a)):
            return bisection(f, bis, b, tol)
        else:
            return bisection(f, a, bis, tol)

def secant(f, x_1, x_0, tol):
    m = (f(x_1) - f(x_0)) / (x_1 - x_0)
    x_new = (m * x_1 - f(x_1)) / m #Point where secant line crosses zero

    if np.abs(f(x_new)) < tol:
        return x_new

    else:
        return secant(f, x_new, x_1, tol)

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
