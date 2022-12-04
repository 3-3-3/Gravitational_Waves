import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Keppler_Equation.keppler import solve_keppler

class Binary_System:
    def __init__(s, ecc, T, R=100, m_1=10, m_2=1, th_p=0, i=0, th_n=0, phi=0, dec=0, ra=0, t=None):
        '''
        ecc: Eccentricity of orbit
        T: Orbital Period
        th_p: Value of th (angular displacement) at pariapse
        i: Inclination of orbital plane from sky tangent plane
        th_n: Value of th at node line
        a: Semi-major axis of orbit, a = a_1 + a_2, where a_1 and a_2 are the
        respective axes for m_1 and m_2
        phi: Oritentation of node lin in sky
        R: Distance to center of mass of binary system
        m_1: Primary
        m_2: Secondary
        dec: Declination of source
        ra: Right ascension of source
        '''

        s.ecc = ecc; s.T = T; s.th_p = th_p; s.i = i; s.th_n = th_n; s.phi = phi; s.R = R

        s.m_1 = m_1; s.m_2 = m_2 #Primary mass, secondary mass
        s.beta = s.m_2 / s.m_1 #Mass Ratio
        s.m_red = s.m_1 * s.m_2 / (s.m_1 + s.m_2) #Reduced mass

        s.G = 1 #Just choose units where G and c are one... for now. Easy enough to update later on
        s.c = 1

        #Semi-major axis of reduced mass orbit
        s.a = np.cbrt(s.G * m_1 * m_2 * T ** 2 / (4 * np.pi ** 2 * s.m_red))

        #Semi-major axes of m_2 and m_1
        s.a_2 = s.a / (1 + s.beta)
        s.a_1 = s.a - s.a_2

        #Semi-minor axes
        s.b_1 = np.sqrt((1 - s.ecc ** 2) * s.a_1 ** 2)
        s.b_2 = np.sqrt((1 - s.ecc ** 2) * s.a_2 ** 2)

        s.H = (4 * s.G ** 2 * s.m_1 * s.m_2) / (s.c ** 4 * s.a * (1 - s.ecc ** 2) * s.R) #Scaling term from Wahlquist

        #Star coordinates; generally given in degrees I believe, but I am converting to radians
        #So that I do not need to convert everytime I call np.cos or np.sin
        s.ra = ra * (np.pi / 180)
        s.dec = dec * (np.pi / 180)

        s.a_hat = np.array([-np.sin(s.ra), np.cos(s.ra), 0]) #Unit vector in increasing right ascencion
        s.d_hat = np.array([-np.sin(s.dec) * np.cos(s.ra), -np.sin(s.dec) * np.sin(s.ra), np.cos(s.dec)])

        #Tensordot (a, b), last arg 0: matrix [[a_xb_x, a_xb_y, a_xb_z], [a_yb_x, a_yb_y, a_yb_z], [a_zb_x, a_zb_y, a_za_z]]
        #Not really sure if this is correct, but doesn't really matter... we can just choose propogation in z
        s.e_plus = np.tensordot(s.a_hat, s.a_hat, 0) - np.tensordot(s.d_hat, s.d_hat, 0)
        s.e_cross = np.tensordot(s.a_hat, s.d_hat, 0) + np.tensordot(s.d_hat, s.a_hat, 0)

        if t == None:
            s.t = np.array(0, 2 * T, 5000)
        else:
            s.t = t


    def A_0(s, th):
        '''
        A_0, as defined in The Doppler Response to Gravitational Waves
        from a Binary Star Source, Wahlquist, 1987
        th: angular displacement
        '''
        return -1 / 2 * (1 + np.cos(s.i) ** 2) * np.cos(2 * (th - s.th_n))

    def B_0(s, th):
        '''
        B_0, as defined the Wahlquist
        '''
        return -np.cos(s.i) * np.sin(2 * (th - s.th_n))

    def A_1(s, th):
        return 1 / 4 * (np.sin(s.i) ** 2 * np.cos(th - s.th_p)) \
            - 1 / 8 * (1 + np.cos(s.i) ** 2) * (5 * np.cos(th - 2 * s.th_n + s.th_p) \
            + np.cos(3 * th - 2 * s.th_n - s.th_p))

    def B_1(s, th):
        return -1 / 4 * np.cos(s.i) * (5 * np.sin(th - 2 * s.th_n + s.th_p) \
            + np.sin(3 * th - 2 * s.th_n - s.th_p))

    def A_2(s, th):
        return 1 / 4 * np.sin(s.i) ** 2 \
            - 1 / 4 * (1 + np.cos(s.i) ** 2) * np.cos(2 * (s.th_n - s.th_p))

    def B_2(s, th):
        return 1 / 2 * np.cos(s.i) * np.sin(2 * s.th_n - s.th_p)

    def h_plus(s, th):
        '''
        Plus polarization, as derived in Wahlquist
        '''
        return s.H * (np.cos(2 * s.phi) * (s.A_0(th) + s.ecc * s.A_1(th) + s.ecc ** 2 * s.A_2(th)) \
            - np.sin(s.phi) * (s.B_0(th) + s.ecc * s.B_1(th) + s.ecc ** 2 * s.B_2(th)))

    def h_cross(s, th):
        '''
        Cross polarization, as derived in Wahlquist
        '''
        return s.H * (np.sin(2 * s.phi) * (s.A_0(th) + s.ecc * s.A_1(th) + s.ecc ** 2 * s.A_2(th)) \
            + np.cos(s.phi) * (s.B_0(th) + s.ecc * s.B_1(th) + s.ecc ** 2 * s.B_2(th)))

    def areal_velocity_1(s):
        def r(th, a_p, ecc_p):
            '''
            th: Parameter th
            a_p: Semi-major axis of orbit
            Returns: r to be used in parametric equation of ellipse
            '''
            return a_p * (1 - ecc_p) * (1 + ecc_p) / (1 + ecc_p * np.cos(th))


        theta = s.th()
        da_1 = np.empty(theta.size - 1)

        for i in range(s.t.size - 1):
            r_ave = (r(theta[i + 1], s.a_1, s.ecc) + r(theta[i], s.a_1, s.ecc)) / 2
            d_theta_d_t = (theta[i + 1] - theta[i]) / (s.t[i + 1] - s.t[i])
            da_1[i] = r_ave ** 2 * d_theta_d_t


        return da_1

    def areal_velocity_2(s):
        def r(th, a_p, ecc_p):
            '''
            th: Parameter th
            a_p: Semi-major axis of orbit
            Returns: r to be used in parametric equation of ellipse
            '''
            return a_p * (1 - ecc_p) * (1 + ecc_p) / (1 + ecc_p * np.cos(th))


        theta = s.th()
        da_2 = np.empty(theta.size - 1)

        for i in range(s.t.size - 2):
            r_ave = (r(theta[i + 1], s.a_2, s.ecc) + r(theta[i], s.a_2, s.ecc)) / 2
            d_theta_d_t = (theta[i + 1] - theta[i]) / (s.t[i + 1] - s.t[i])
            da_2[i] = r_ave ** 2 * d_theta_d_t


        return da_2

    def animate_orbits(s):
        '''
        Visualize star orbit (in the orbital plane)
        '''
        def r(th, a_p, ecc_p):
            '''
            th: Parameter th
            a_p: Semi-major axis of orbit
            Returns: r to be used in parametric equation of ellipse
            '''
            return a_p * (1 - ecc_p) * (1 + ecc_p) / (1 + ecc_p * np.cos(th))

        def animate(f, r_1, r_2, h_p, h_c, orbit_1, orbit_2, line_1, line_2):
            '''
            f: Frames being used as a proxy for parameter
            r_1: List of position vectors at each theta for orbit 1
            r_2: List of position vectors at each theta for orbit 2
            h_p: List of magnitude of h-plus at various angles
            h_c: List of magnitude of h-cross at various angles
            orbit_1: artist for first orbit
            orbit_2: artist for second orbit
            line_1: artist for h_p
            line_2: artist for h_c
            '''
            orbit_1.set_data(r_1[0, :f], r_1[1, :f])
            orbit_2.set_data(r_2[0, :f], r_2[1, :f])

            line_1.set_data(s.t[:f], h_p[:f])
            line_2.set_data(s.t[:f], h_c[:f])

            return [orbit_1, orbit_2, line_1, line_2]

        th = s.th() #theta array as a function of time

        r_1 = r(th, s.a_1, s.ecc) * np.array([np.cos(th - s.th_p), np.sin(th - s.th_p)])
        r_2 = r(th, s.a_2, s.ecc) * np.array([-np.cos(th - s.th_p), -np.sin(th - s.th_p)])

        h_p = s.h_plus(th)
        h_c = s.h_cross(th)

        fig, axes = plt.subplots(2)

        #Frame about the larger orbit
        axes[0].set_xlim([-1.1 * max(s.a_1, s.a_2) * (1 - s.ecc), 1.1 * max(s.a_1, s.a_2) * (1 + s.ecc)])
        axes[0].set_ylim([-1.1 * max(s.b_1, s.b_2), 1.1 * max(s.b_1, s.b_2)])

        axes[1].set_xlim(th.min(), th.max())
        axes[1].set_ylim(min(1.1 * min(h_p.min(), h_c.min()), 0.9 * min(h_p.min(), h_c.min())),
                         max(1.1 * max(h_p.max(), h_c.max()), 0.9 * max(h_p.max(), h_c.max())))

        #Artist objects returned from plot function and to be used in animation
        orbit_1, = axes[0].plot(r_1[0, :1], r_1[1, :1], 'o-', color='pink', \
                        label='Primary', mfc='r', mec='r', markersize=10, markevery=[-1])

        orbit_2, = axes[0].plot(r_2[0, :1], r_2[1, :1], 'o-', color='purple', \
                        label='Secondary', mfc='r', mec='r', markersize=10, markevery=[-1])

        line_1, = axes[1].plot(s.t[:1], h_p[:1], color='green', label='h-plus')
        line_2, = axes[1].plot(s.t[:1], h_c[:1], color='blue', label='h-cross')

        axes[0].legend(loc='upper right')
        axes[1].legend(loc='upper right')

        axes[0].grid()
        axes[1].grid()

        ani = FuncAnimation(fig, animate, frames=th.size, interval=10, \
                            fargs=[r_1, r_2, h_p, h_c, orbit_1, orbit_2, line_1, line_2])

        plt.show()

    def psi(s):
        return solve_keppler(s.t, s.T, s.ecc) #Solve for psi at time samples

    def th(s):
        #x = np.cos(np.sqrt((1 + s.ecc) / (1 - s.ecc)) * np.tan(s.psi(t / 2)))
        #y = np.sin(np.sqrt((1 + s.ecc) / (1 - s.ecc)) * np.tan(s.psi(t / 2)))
        return 2 * np.arctan(np.sqrt((1 + s.ecc) / (1 - s.ecc)) * np.tan(s.psi() / 2)) #theta as a function of time
