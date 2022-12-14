import numpy as np

class OdeInt:
    def __init__(s, de_tau, e_0, tau_0, dtau_0, tau_f, a_tol=0, r_tol=0):
        s.e_0 = e_0
        s.tau = tau_0
        s.tau_0 = tau_0
        s.tau_f = tau_f
        s.a_tol = a_tol
        s.r_tol = r_tol
        s.dtau_0 = dtau_0
        s.e_n = np.empty((tau_0 - tau_f) / dtau_0)
        s.tau = np.empty(s.e_n.size)
        s.count = 0
        s.stepper = Adaptive_Stepper(s.de_dtau, s.e_0, s.tau_0, s.dtau_0, a_tol=s.a_tol, r_tol=s.r_tol)

    def integrate(s):
        while s.tau[count] < s.tau_f:
            if s.e_n.size <= s.count:
                s.e_n = np.resize(s.e_n.size * 2)
                s.tau = np.resize(s.e_n.size * 2)

        s.stepper.step()
        s.e_n[count] = s.stepper.e_n
        s.tau[count] = s.stepper.tau

        count += 1

class Adaptive_Stepper:
    '''
    Runge-Kutta stpper class
    '''
    def __init__(s, de_dtau, e_0, tau_0, dtau_0, a_tol=0, r_tol=0):
        '''
        de_dtau: derivative, function of tau and e
        e_0: initial Eccentricity
        tau_0: initial time
        dtau_0: Guess for step size
        a_tol: absolute tolerance
        r_tol: relative tolerance
        '''
        s.e_n = e_0
        s.tau = tau_0
        s.a_tol = a_tol
        s.r_tol = r_tol
        s.dtau = dtau_0
        s.de_dtau = de_dtau

        #Stepper coefficients
        s.c = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
        s.b_i_5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]) #coefficients for 5th order routine
        s.b_i_6 = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]) #coefficients for 6th order routine
        s.a = np.array([[1/5, 0, 0, 0, 0, 0],
                      [3 / 40, 9 / 40, 0, 0, 0, 0],
                      [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
                      [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
                      [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0],
                      [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]]) #Coefficients to feed into k_{n+1}


    def step(s):
        #O(dt^5) and O(dt^6) approximations
        e_5 = 0; e_6 = 0
        delta = 0 #First order ODE so delta and scale are scalar
        scale = 0

        #Evaluate each k
        k = np.empty(6)
        k[0] = s.dtau * s.de_dtau(s.tau, s.e_n) #Euler step
        k[1] = s.dtau * s.de_dtau(s.tau + s.c[1] * s.tau, s.e_n + np.dot(s.a[0], k)) #Other steps
        k[2] = s.dtau * s.de_dtau(s.tau + s.c[2] * s.tau, s.e_n + np.dot(s.a[1], k))
        k[3] = s.dtau * s.de_dtau(s.tau + s.c[3] * s.tau, s.e_n + np.dot(s.a[2], k))
        k[4] = s.dtau * s.de_dtau(s.tau + s.c[4] * s.tau, s.e_n + np.dot(s.a[3], k))
        k[5] = s.dtau * s.de_dtau(s.tau + s.c[5] * s.tau, s.e_n + np.dot(s.a[4], k))

        e_5 = s.e_n + np.dot(k, s.b_i_5[:-1]); e_6 = s.e_n + np.dot(k, s.b_i_6[:-1]) #Calculate O(h^6) approximation and embedded O(h^5) approximation


        if np.isnan(e_5):
            raise ValueError
        delta = e_6 - e_5 #Error estimate
        scale = s.a_tol + max(s.e_n, e_6) * s.r_tol #Error we are shooting for
        err = np.sqrt((delta / scale) ** 2) #Ratio of step error to desired error, scale

        print(f'e_n: {s.e_n}, tau: {s.tau}, dtau: {s.dtau}, e_5: {e_5}, e_6: {e_6}, error: {err}')


        #Return True if step size accepted, False otherwise
        if err <= 1:
            s.e_n = e_6
            s.tau = s.tau + s.dtau

            #update step size
            #As err < 1, the new step size will be larger
            s.dtau = s.dtau * (1 / np.abs(err)) ** (1 / 5)


        else:
            #Because err > 1, it will be smaller; 0.95 is a safety factor
            s.dtau = 0.999999 * s.dtau * (1 / err) ** (1 / 5)
            #Try again
            s.step()
