import numpy as np


#From Numerical Recipes
class Trap:
    def __init__(s, f, a, b, args=[]):
        '''
        f: function to be integrated
        a,b: limits of integration
        '''
        s.f = f
        s.a = a; s.b = b
        s.s = 0 #sum
        s.n = 0 #Level of approximation
        s.args=args

    def next(s):
        s.n += 1
        if s.n == 1:
            s.s = 0.5 * (s.b - s.a) * (s.f(s.a, *s.args) + s.f(s.b, *s.args))
            return s.s

        else:
            points = s.n - 1
            interval = (s.b - s.a) / points
            x = s.a + interval
            sum = 0

            for i in range(points):
                sum += s.f(x, *s.args)
                x += interval

            s.s = 0.5 * (s.s + (s.b - s.a) * sum / points)
            return s.s

def trap_quad(f, a, b, args=[], eps=1e-5):
    t = Trap(f, a, b, args=args)
    s = 0; old_s = 0
    JMAX = int(10e5)

    for j in range(JMAX):
        s = t.next()

        if j > 5:
            if abs(s - s_old) < eps * abs(s_old) or (s == 0 and s_old == 0):
                return s

        s_old = s

    raise Exception('Too many steps in q_trap')
