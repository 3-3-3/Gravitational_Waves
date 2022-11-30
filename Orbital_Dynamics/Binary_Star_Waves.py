import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Binary_System:
    def __init__(s, th_p, ecc, i, th_n, a, phi, R, m_1, m_2, dec, ra):
        '''
        th_p: Value of th (angular displacement) at pariapse
        ecc: Eccentricity of orbit
        i: Inclination of orbital plane from sky tangent plane
        th_n: Value of th at node line
        a: Semi-major axis of orbit, a = a_1 + a_2, where a_1 and a_2 are the
        respective axes for m_1 and m_2
        phi: Oritentation of node lin in sky
        R: Position of center of mass of binary system
        m_1: Primary
        m_2: Secondary
        dec: Declination of source
        ra: Right ascension of source
        '''

        s.th_p = th_p; s.ecc = ecc; s.i = i; s.th_n = th_n; s.phi = phi; s.R = R
        s.a = a #Semi-major radius of reduced mass, one body orbit

        s.m_1 = m_1; s.m_2 = m_2 #Primary mass, secondary mass
        s.beta = s.m_2 / s.m_1 #Mass Ratio

        #Semi-major axes of m_2 and m_1
        s.a_2 = s.a / (1 + s.beta)
        s.a_1 = s.a - s.a_2

        #Semi-minor axes
        s.b_1 = np.sqrt((1 - s.ecc ** 2) * s.a_1 ** 2)
        s.b_2 = np.sqrt((1 - s.ecc ** 2) * s.a_2 ** 2)

        s.G = 1
        s.c = 1
        s.H = (4 * s.G ** 2 * s.m_1 * s.m_2) / (s.c ** 4 * s.a * (1 - s.ecc ** 2) * s.R)

        s.ra = ra
        s.dec = dec

        s.a_hat = np.array([-np.sin(s.ra), np.cos(s.ra), 0]) #Unit vector in increasing right ascencion
        s.d_hat = np.array([-np.sin(s.dec) * np.cos(s.ra), -np.sin(s.dec) * np.sin(s.ra), np.cos(s.dec)])
        #Tensordot (a, b), last arg 0: matrix [[a_xb_x, a_xb_y, a_xb_z], [a_yb_x, a_yb_y, a_yb_z], [a_zb_x, a_zb_y, a_za_z]]
        s.e_plus = np.tensordot(s.a_hat, s.a_hat, 0) - np.tensordot(s.d_hat, s.d_hat, 0)
        s.e_cross = np.tensordot(s.a_hat, s.d_hat, 0) + np.tensordot(s.d_hat, s.a_hat, 0)


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

    def animate_h_plus(s, th):
        def animate(t, h_p, line):
            line.set_data(th[:t], h_p[:t])
            return [line]


        h_p = s.h_plus(th)
        fig, ax = plt.subplots()
        line, = ax.plot(th[:1], h_p[:1], color='green', label='h-plus')

        ax.set_xlim(th.min(), th.max())
        ax.set_ylim(h_p.min(), h_p.max())

        ani = FuncAnimation(fig, animate, frames=th.size, interval=25, \
                            fargs=[h_p, line])

        plt.show()

    def h_cross(s, th):
        '''
        Cross polarization, as derived in Wahlquist
        '''
        return s.H * (np.sin(2 * s.phi) * (s.A_0(th) + s.ecc * s.A_1(th) + s.ecc ** 2 * s.A_2(th)) \
            + np.cos(s.phi) * (s.B_0(th) + s.ecc * s.B_1(th) + s.ecc ** 2 * s.B_2(th)))

    def animate_orbits(s, th):
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

        def animate(t, r_1, r_2, h_p, h_c, orbit_1, orbit_2, line_1, line_2):
            '''
            t: Frames being used as a proxy for parameter th
            r_1: List of position vectors at each time for orbit 1
            r_2: List of position vectors at each time for orbit 2
            line_1: artist for first orbit
            line_2: artist for second
            '''
            orbit_1.set_data(r_1[0, :t], r_1[1, :t])
            orbit_2.set_data(r_2[0, :t], r_2[1, :t])

            line_1.set_data(th[:t], h_p[:t])
            line_2.set_data(th[:t], h_c[:t])

            return [orbit_1, orbit_2, line_1, line_2]

        r_1 = r(th, s.a_1, s.ecc) * np.array([np.cos(th - s.th_p), np.sin(th - s.th_p)])
        r_2 = r(th, s.a_2, s.ecc) * np.array([-np.cos(th - s.th_p), -np.sin(th - s.th_p)])

        h_p = s.h_plus(th)
        plt.plot(th,h_p)
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

        line_1, = axes[1].plot(th[:1], h_p[:1], color='green', label='h-plus')
        line_2, = axes[1].plot(th[:1], h_c[:1], color='blue', label='h-cross')

        axes[0].legend(loc='upper right')
        axes[1].legend(loc='upper right')

        axes[0].grid()
        axes[1].grid()

        ani = FuncAnimation(fig, animate, frames=th.size, interval=15, \
                            fargs=[r_1, r_2, h_p, h_c, orbit_1, orbit_2, line_1, line_2])
        plt.show()
