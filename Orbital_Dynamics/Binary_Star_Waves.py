import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from Keppler_Equation.keppler import solve_keppler
from quad.quad_routines import trap_quad
import scipy.constants as pc

class Binary_System:
    def __init__(s, ecc, T, R=100, m_1=10, m_2=6, th_p=0, i=0, th_n=0, phi=0, dec=0, ra=0, t=None, min_a=0):
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

        #Choose units of distance in au, time in years, and mass in solar masses
        s.G = pc.G * (1 / 1.496e11) ** 3 * (1.989e30) * (60 * 60 * 24 * 365.25) ** 2
        s.c = pc.c  * (1 / 1.496e11) * (60 * 60 * 24 * 365.25) #c in au per year

        #Semi-major axis of reduced mass orbit
        s.a = np.cbrt(s.G * m_1 * m_2 * T ** 2 / (4 * np.pi ** 2 * s.m_red))

        #Semi-major axes of m_2 and m_1
        s.a_2 = s.a / (1 + s.beta)
        s.a_1 = s.a - s.a_2

        #Semi-minor axes
        s.b_1 = np.sqrt((1 - s.ecc ** 2) * s.a_1 ** 2)
        s.b_2 = np.sqrt((1 - s.ecc ** 2) * s.a_2 ** 2)

        #c_0 as defined in Peters; chose so a(e_0) = a_0
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
            s.t = np.linspace(0, 2 * s.T, 2000)
        else:
            s.t = t

        plt.style.use('ggplot')

        s.min_a = min_a
        s.count = 0
        s.ecc_array = np.empty(1e10)
        s.a_array = np.empty(1e10)
        s.T_array = np.empty(1e10)


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

    def animate_orbits(s, out_file=None):
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

        fig, axes = plt.subplots(2,figsize=(7,7))

        #Frame about the larger orbit
        axes[0].set_xlim([1.1 * min(r_1[0].min(), r_2[0].min()), 1.1 * max(r_1[0].max(), r_2[0].max())])
        axes[0].set_ylim([-1.1 * max(s.b_1, s.b_2), 1.1 * max(s.b_1, s.b_2)])

        axes[1].set_xlim(s.t.min(), s.t.max())
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

        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')

        axes[1].set_xlabel('t')
        axes[1].set_ylabel('Wave Amplitude')

        #Choose interval to complete an orbit in 10s
        n_points = s.t.size / (s.t.max() / s.T)
        interval = (10 ** 3) / (n_points)  #interval measured in milliseconds

        ani = FuncAnimation(fig, animate, frames=th.size, interval=interval, \
                            fargs=[r_1, r_2, h_p, h_c, orbit_1, orbit_2, line_1, line_2])

        s_ecc = r'$\epsilon=$' + str(round(s.ecc, 2))
        s_phi = r'$\phi=$' + str(round(s.phi, 2))
        s_i = r'$i=$' + str(round(s.i, 2))
        s_m_1 = r'$m_1=$' + str(round(s.m_1, 2))
        s_m_2 = r'$m_2=$' + str(round(s.m_2, 2))

        axes[0].set_title(f'Orbit Waves: {s_ecc}, {s_phi}, {s_i}, {s_m_1}, {s_m_2}')

        if out_file:
            writergif = PillowWriter(fps=240)
            ani.save(out_file, writer=writergif)

        plt.show()

    def psi(s):
        '''
        Numerically solves the Keppler equation at each discrete time in s.t
        '''
        return solve_keppler(s.t, s.T, s.ecc) #Solve for psi at time samples

    def th(s):
        '''
        Uses s.psi to calculate theta at each discrete time in s.t
        '''
        return 2 * np.arctan(np.sqrt((1 + s.ecc) / (1 - s.ecc)) * np.tan(s.psi() / 2)) #theta as a function of time

    def wave_plots(s, ax):
        '''
        Function to plot waveforms, to be used in iteration methods
        '''
        h_p = s.h_plus(s.th())
        h_c = s.h_cross(s.th())


        ax.plot(s.t, h_p, color='Green', label='h-plus')
        ax.plot(s.t, h_c, color='Blue', label='h-cross')



    def iterate_ecc(s, iterations=9, show=True, save=False):
        '''
        Function to plot wave forms
        For a veriety of eccentricities
        And plot them in the same figure
        '''
        fig, axes = plt.subplots(3, int(np.ceil(iterations / 3)), sharey=True, sharex=True)
        plt.style.use('ggplot')

        inc = 1 / (iterations)
        for i in range(iterations):
            s.ecc = i * inc
            axes[int(i / 3), i % 3].set_title(r'$\epsilon = $' + str(round(s.ecc, 2)))
            s.wave_plots(axes[int(i / 3), i % 3])

        s_ecc = r'$\epsilon$'
        s_phi = r'$\phi=$' + str(round(s.phi, 2))
        s_i = r'$i=$' + str(round(s.i, 2))
        s_m_1 = r'$m_1=$' + str(round(s.m_1, 2))
        s_m_2 = r'$m_2=$' + str(round(s.m_2, 2))

        fig.suptitle(f'Iterate {s_ecc}: {s_phi}, {s_i}, {s_m_1}, {s_m_2}')

        if save: plt.savefig('Eccentricity_Plots.png')

        if show: plt.show()

    def iterate_i(s, iterations=9, show=True, save=False):
        '''
        Function to plot wave forms
        For a veriety of inclinations
        And plot them in the same figure
        '''
        fig, axes = plt.subplots(3, int(np.ceil(iterations / 3)), sharey=True, sharex=True)
        plt.style.use('ggplot')

        inc = 2 * np.pi / (iterations)
        for i in range(iterations):
            s.i = i * inc
            axes[int(i / 3), i % 3].set_title(r'$i = $' + str(round(s.i, 2)))
            s.wave_plots(axes[int(i / 3), i % 3])

        s_ecc = r'$\epsilon$' + str(round(s.ecc, 2))
        s_phi = r'$\phi=$' + str(round(s.phi, 2))
        s_i = r'$i$'
        s_m_1 = r'$m_1=$' + str(round(s.m_1, 2))
        s_m_2 = r'$m_2=$' + str(round(s.m_2, 2))

        fig.suptitle(f'Iterate {s_i}: {s_phi}, {s_i}, {s_m_1}, {s_m_2}')

        if save: plt.savefig('Inclination_Plots.png')

        if show: plt.show()

    def iterate_phi(s, iterations=9, show=True, save=False):
        '''
        Function to plot wave forms
        For a veriety of orientations
        And plot them in the same figure
        '''
        fig, axes = plt.subplots(3, int(np.ceil(iterations / 3)), sharey=True, sharex=True)

        inc = 2 * np.pi / (iterations)
        for i in range(iterations):
            s.th_n = i * inc
            s.phi = i * inc #Incriment both the physical angle phi, and the arbitrary
                            #angle th_n, which is the choice of where the
                            #nodeline is oriented in the binary system
            axes[int(i / 3), i % 3].set_title(r'$\phi = $' + str(round(s.th_n / np.pi, 2)) + r'$\pi$')
            s.wave_plots(axes[int(i / 3), i % 3])

        s_ecc = r'$\epsilon$' + str(round(s.ecc),2)
        s_phi = r'$\phi$'
        s_i = r'$i=$' + str(round(s.i, 2))
        s_m_1 = r'$m_1=$' + str(round(s.m_1, 2))
        s_m_2 = r'$m_2=$' + str(round(s.m_2, 2))

        fig.suptitle(f'Iterate {s_phi}: {s_ecc}, {s_i}, {s_m_1}, {s_m_2}')

        if save: plt.savefig(r'Node_Orientation_Plots.png')

        if show: plt.show()

    def de(s,T_between=1):
        '''
        change in eccentricity per orbit as found by approximating Peters eq. 5.7
        T_between: number of periods to approximate with
        '''
        return -304 / 15 * s.ecc * (s.G ** 3 * s.m_1 * s.m_2 * (s.m_1 + s.m_2)) \
                / (s.c ** 5 * s.a ** 4 * (1 - s.ecc ** 2) ** (5/2)) \
                * (1 + (121 / 304) * s.ecc ** 2) * s.T * T_between

    def da(s,T_between=1):
        '''
        change in semi-major axis per orbit as found by approximating Peters eq. 5.6
        T_between: number of periods to approximate with
        '''
        return -64 / 5 * (s.G ** 3 * s.m_1 * s.m_2 * (s.m_1 + s.m_2)) \
                            / (s.c ** 5 * s.a ** 3 * (1 - s.ecc ** 2) ** (7/2)) \
                            * (1 + 73 / 24 * s.ecc ** 2 + 37 / 96 * s.ecc ** 4) * s.T * T_between

    def da_de(s,T_between=1):
        '''
        change in semi-major axis per change in eccentricity as approximated by Peters eq. 5.8
        T_between: number of periods to approximate with
        '''
        return 12 / 19 * (s.a / s.ecc) \
                    * (1 + (73 / 24) * s.ecc ** 2 + (37 / 96) * s.ecc ** 4) \
                    / ((1 - s.ecc ** 2) * (1 + (121 / 304) * s.ecc ** 2)) * T_between

    def dE(s,T_between=1):
        '''
        change in energy of system per orbit as approximated by Peters eq. 5.4
        T_between: number of periods to approximate with
        '''
        return -32 / 5 * (s.G ** 4 * s.m_1 ** 2 * s.m_2 ** 2 * (s.m_1 + s.m_2)) \
                        / (s.c ** 5 * s.a ** 5 * (1 - s.ecc ** 2) ** (7 / 2)) \
                        * (1 + 73 / 24 * s.ecc ** 2 + 37 / 96 * s.ecc ** 4) * s.T * T_between

    def step(s, calc_every=1e4):
        if s.count == s.t_array.size:
            s.h_array = np.arange(2 * h_array.size)
            s.ecc_array.resize(2 * ecc_array.size)
            s.a_array.resize(2 * a_array.size)
            s.T_array.resize(2 * T_array.size)


        s.ecc = s.ecc + s.de(T_between=calc_every)
        s.a = s.a + s.da(T_between=calc_every)
        s.T = np.sqrt((4 * np.pi ** 2 * s.m_red * s.a ** 3) / (s.G * s.m_1 * s.m_2))

        if s.a <= s.min_a:
            return True

        s.count += 1
        if s.count % 10000 == 0:
            print(f'Step: {s.count}')
        s.ecc_array[count] = s.ecc; s.a_array[count] = s.a; s.T_array[count] = s.T
        return False

    def orbit_decay(s):
        if s.step():
            return (s.T_array, s.ecc_array, s.a_array)
        return s.orbit_dec()





if __name__ == '__main__':
    b = Binary_System(0.6, 1)
    b.animate_orbits(out_file='orbit.gif')
