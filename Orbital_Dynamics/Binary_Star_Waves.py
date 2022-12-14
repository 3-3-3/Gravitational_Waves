import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from Keppler_Equation.keppler import solve_keppler
import scipy.constants as pc
from scipy.integrate import ode

def to_au(L):
    '''
    meters to austronomical units
    '''
    return L / (pc.au)

def to_days(t):
    '''
    Geometerized time to seconds
    '''
    return t / (24 * 60 * 60 * pc.c)

def to_sm(M):
    '''
    Geometerized mass to solar mass
    '''
    return M * pc.c ** 2 / (pc.G) / 1.989e30

class Binary_System:
    def __init__(s, ecc, T, R=100, m_1=10, m_2=6, th_p=0, i=0, th_n=0, phi=0, dec=0, ra=0, t=None, a_min=0):
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
        s.ecc_0 = ecc;
        s.ecc = ecc; s.T = T; s.th_p = th_p; s.i = i; s.th_n = th_n; s.phi = phi; s.R = R
        s.a_min = a_min

        s.m_1 = m_1; s.m_2 = m_2 #Primary mass, secondary mass
        s.beta = s.m_2 / s.m_1 #Mass Ratio
        s.m_red = s.m_1 * s.m_2 / (s.m_1 + s.m_2) #Reduced mass

        s.G = 1 #Just choose units where G and c are one... for now. Easy enough to update later on
        s.c = 1

        #Semi-major axis of reduced mass orbit
        s.a = np.cbrt(s.G * m_1 * m_2 * T ** 2 / (4 * np.pi ** 2 * s.m_red))
        s.a_0 = np.cbrt(s.G * m_1 * m_2 * T ** 2 / (4 * np.pi ** 2 * s.m_red))

        #Semi-major axes of m_2 and m_1
        s.a_2 = s.a / (1 + s.beta)
        s.a_1 = s.a - s.a_2

        #Semi-minor axes
        s.b_1 = np.sqrt((1 - s.ecc ** 2) * s.a_1 ** 2)
        s.b_2 = np.sqrt((1 - s.ecc ** 2) * s.a_2 ** 2)

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
            s.t = np.linspace(0, s.T, 1000)
        else:
            s.t = t

        plt.style.use('ggplot')




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

    def H(s):
        '''
        Scaling factor in Wahlquist
        '''
        return (4 * s.G ** 2 * s.m_1 * s.m_2) / (s.c ** 4 * s.a * (1 - s.ecc ** 2) * s.R)

    def h_plus(s, th):
        '''
        Plus polarization, as derived in Wahlquist
        '''
        return s.H() * (np.cos(2 * s.phi) * (s.A_0(th) + s.ecc * s.A_1(th) + s.ecc ** 2 * s.A_2(th)) \
            - np.sin(2 * s.phi) * (s.B_0(th) + s.ecc * s.B_1(th) + s.ecc ** 2 * s.B_2(th)))

    def h_cross(s, th):
        '''
        Cross polarization, as derived in Wahlquist
        '''
        return s.H() * (np.sin(2 * s.phi) * (s.A_0(th) + s.ecc * s.A_1(th) + s.ecc ** 2 * s.A_2(th)) \
            + np.cos(2 * s.phi) * (s.B_0(th) + s.ecc * s.B_1(th) + s.ecc ** 2 * s.B_2(th)))

    def areal_velocity_1(s):
        def r(th, a_p, ecc_p):
            '''
            th: Parameter th
            a_p: Semi-major axis of orbit
            Returns: r to be used in parametric equation of ellipse
            '''
            return a_p * (1 - ecc_p) * (1 + ecc_p) / (1 + ecc_p * np.cos(th))


        theta = s.th(s.t)
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


        theta = s.th(s.t)
        da_2 = np.empty(theta.size - 1)

        for i in range(s.t.size - 2):
            r_ave = (r(theta[i + 1], s.a_2, s.ecc) + r(theta[i], s.a_2, s.ecc)) / 2
            d_theta_d_t = (theta[i + 1] - theta[i]) / (s.t[i + 1] - s.t[i])
            da_2[i] = r_ave ** 2 * d_theta_d_t


        return da_2

    def animate_orbits(s, square=True, out_file=None):
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
            orbit_1.set_data(to_au(r_1[0, :f]), to_au(r_1[1, :f]))
            orbit_2.set_data(to_au(r_2[0, :f]), to_au(r_2[1, :f]))

            line_1.set_data(to_days(s.t[:f]), h_p[:f])
            line_2.set_data(to_days(s.t[:f]), h_c[:f])

            return [orbit_1, orbit_2, line_1, line_2]

        th = s.th(s.t) #theta array as a function of time

        r_1 = r(th, s.a_1, s.ecc) * np.array([np.cos(th - s.th_p), np.sin(th - s.th_p)])
        r_2 = r(th, s.a_2, s.ecc) * np.array([-np.cos(th - s.th_p), -np.sin(th - s.th_p)])

        h_p = s.h_plus(th)
        h_c = s.h_cross(th)

        fig, axes = plt.subplots(2,figsize=(8,7))

        #Frame about the larger orbit
        if square:
            axes[0].set_aspect('equal')
            axes[0].set_xlim([1.1 * min(to_au(r_1[0].min()), to_au(r_2[0].min())), 1.1 * max(to_au(r_1[0].max()), to_au(r_2[0].max()))])
            axes[0].set_ylim([-1.1 * to_au(max(s.b_1, s.b_2)), 1.1 * to_au(max(s.b_1, s.b_2))])
        else:
            axes[0].set_xlim([1.1 * min(to_au(r_1[0].min()), to_au(r_2[0].min())), 1.1 * max(to_au(r_1[0].max()), to_au(r_2[0].max()))])
            axes[0].set_ylim([-1.1 * to_au(max(s.b_1, s.b_2)), 1.1 * to_au(max(s.b_1, s.b_2))])


        axes[1].set_xlim(to_days(s.t.min()), to_days(s.t.max()))
        axes[1].set_ylim(min(1.1 * min(h_p.min(), h_c.min()), 0.9 * min(h_p.min(), h_c.min())),
                         max(1.1 * max(h_p.max(), h_c.max()), 0.9 * max(h_p.max(), h_c.max())))

        #Artist objects returned from plot function and to be used in animation
        orbit_1, = axes[0].plot(to_au(r_1[0, :1]), to_au(r_1[1, :1]), 'o-', color='pink', \
                        label='Primary', mfc='r', mec='r', markersize=10, markevery=[-1])

        orbit_2, = axes[0].plot(to_au(r_2[0, :1]), to_au(r_2[1, :1]), 'o-', color='purple', \
                        label='Secondary', mfc='r', mec='r', markersize=10, markevery=[-1])

        line_1, = axes[1].plot(s.t[:1], h_p[:1], color='green', label=r'$h_+$')
        line_2, = axes[1].plot(s.t[:1], h_c[:1], color='blue', label=r'$h_x$')

        axes[0].legend(loc='upper right')
        axes[1].legend(loc='upper right')

        axes[0].set_xlabel('x (au)')
        axes[0].set_ylabel('y (au)')

        axes[1].set_xlabel('t (days)')
        axes[1].set_ylabel('Wave Amplitude')

        #Choose interval to complete an orbit in 10s
        n_points = s.t.size / (s.t.max() / s.T)

        if 2000 / n_points > 1:
            interval = (2000) / (n_points)  #interval measured in milliseconds
        else:
            interval = 1 #Interval cannot be less than 1

        print(interval)
        print(n_points)

        ani = FuncAnimation(fig, animate, frames=th.size, interval=interval, \
                            fargs=[r_1, r_2, h_p, h_c, orbit_1, orbit_2, line_1, line_2], repeat=True)

        s_ecc = r'$\epsilon=$' + str(round(s.ecc, 2))
        s_phi = r'$\phi=$' + str(round(s.phi, 2))
        s_i = r'$i=$' + str(round(s.i, 2))
        s_m_1 = r'$m_1=$' + str(round(to_sm(s.m_1), 2)) + r'$M_{\odot}$'
        s_m_2 = r'$m_2=$' + str(round(to_sm(s.m_2), 2)) + r'$M_{\odot}$'

        axes[0].set_title(f'Orbit Waves: {s_ecc}, {s_phi}, {s_i}, {s_m_1}, {s_m_2}')

        if out_file:
            writergif = PillowWriter(fps=400)
            ani.save(out_file, writer=writergif)

        plt.show()

    def psi(s, t):
        '''
        Numerically solves the Keppler equation at each discrete time in s.t
        '''
        return solve_keppler(t, s.T, s.ecc) #Solve for psi at time samples

    def th(s,t=None):
        '''
        Uses s.psi to calculate theta at each discrete time in a given time array t
        '''
        if type(t) == type(None):
            t = s.t
        return 2 * np.arctan(np.sqrt((1 + s.ecc) / (1 - s.ecc)) * np.tan(s.psi(t) / 2)) #theta as a function of time

    def wave_plots(s, ax):
        '''
        Function to plot waveforms, to be used in iteration methods
        '''
        h_p = s.h_plus(s.th(s.t))
        h_c = s.h_cross(s.th(s.t))


        ax.plot(to_days(s.t), h_p, color='Green', label=r'$h_+$')
        ax.plot(to_days(s.t), h_c, color='Blue', label=r'$h_x$')



    def iterate_ecc(s, iterations=9, show=True, save=False):
        '''
        Function to plot wave forms
        For a veriety of eccentricities
        And plot them in the same figure
        '''
        fig, axes = plt.subplots(3, int(np.ceil(iterations / 3)), figsize=(8,8), sharey=True, sharex=True)
        plt.style.use('ggplot')

        inc = 1 / (iterations)
        for i in range(iterations):
            s.ecc = i * inc
            axes[int(i / 3), i % 3].set_title(r'$\epsilon = $' + str(round(s.ecc, 2)))
            s.wave_plots(axes[int(i / 3), i % 3])
            axes[int(i / 3), i % 3].legend(loc='upper left')

        s_ecc = r'$\epsilon$'
        s_phi = r'$\phi=$' + str(round(s.phi, 2))
        s_i = r'$i=$' + str(round(s.i, 2))
        s_m_1 = r'$m_1=$' + str(round(to_sm(s.m_1), 2)) + r'$M_{\odot}$'
        s_m_2 = r'$m_2=$' + str(round(to_sm(s.m_2), 2)) + r'$M_{\odot}$'

        fig.suptitle(f'Iterate {s_ecc}: {s_phi}, {s_i}, {s_m_1}, {s_m_2}')

        if save: plt.savefig('Eccentricity_Plots.png')

        if show: plt.show()

    def iterate_i(s, show=True, save=False):
        '''
        Function to plot wave forms
        For a veriety of inclinations
        And plot them in the same figure
        '''
        iterations = 9
        fig, axes = plt.subplots(3, int(np.ceil(iterations / 3)), figsize=(8,8), sharey=True, sharex=True)
        plt.style.use('ggplot')

        inc = 2 * np.pi / (iterations - 1)
        labels = [r'0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$', \
                  r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$', r'$2\pi$']
        for i in range(iterations):
            s.i = i * inc
            axes[int(i / 3), i % 3].set_title(r'$i = $' + labels[i])
            s.wave_plots(axes[int(i / 3), i % 3])
            axes[int(i / 3), i % 3].legend(loc='upper center')

        s_ecc = r'$\epsilon=$' + str(round(s.ecc, 2))
        s_phi = r'$\phi=$' + str(round(s.phi, 2))
        s_i = r'$i$'
        s_m_1 = r'$m_1=$' + str(round(to_sm(s.m_1), 2)) + r'$M_{\odot}$'
        s_m_2 = r'$m_2=$' + str(round(to_sm(s.m_2), 2)) + r'$M_{\odot}$'

        fig.suptitle(f'Iterate {s_i}: {s_phi}, {s_ecc}, {s_m_1}, {s_m_2}')

        if save: plt.savefig('Inclination_Plots.png')

        if show: plt.show()

    def iterate_phi(s, show=True, save=False):
        '''
        Function to plot wave forms
        For a veriety of orientations
        And plot them in the same figure
        '''
        iterations = 9
        fig, axes = plt.subplots(3, int(np.ceil(iterations / 3)), figsize=(8,8), sharey=True, sharex=True)

        inc = 2 * np.pi / (iterations-1)
        labels = [r'0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$', \
                  r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$', r'$2\pi$']
        for i in range(iterations):
            #s.th_n = i * inc
            s.phi = i * inc #Incriment both the physical angle phi, and the arbitrary
                            #angle th_n, which is the choice of where the
                            #nodeline is oriented in the binary system
            axes[int(i / 3), i % 3].set_title(r'$\phi = $' + labels[i])
            s.wave_plots(axes[int(i / 3), i % 3])
            axes[int(i / 3), i % 3].legend(loc='upper center')

        s_ecc = r'$\epsilon=$' + str(round(s.ecc,2))
        s_phi = r'$\phi$'
        s_i = r'$i=$' + str(round(s.i, 2))
        s_m_1 = r'$m_1=$' + str(round(to_sm(s.m_1), 2)) + r'$M_{\odot}$'
        s_m_2 = r'$m_2=$' + str(round(to_sm(s.m_2), 2)) + r'$M_{\odot}$'

        fig.suptitle(f'Iterate {s_phi}: {s_ecc}, {s_i}, {s_m_1}, {s_m_2}')

        if save: plt.savefig(r'Node_Orientation_Plots.png')

        if show: plt.show()


    #Check for scaling factor
    def a_e(s, ecc):
        '''
        semi-major axis as a function of eccentricity
        As given by Maggoire 4.128
        '''
        def g(e):
            return e ** (12 / 19) / (1 - e ** 2) * (1 + 121 / 304 * e ** 2) ** (870 / 2299)

        return s.a_0 * g(ecc) / g(s.ecc_0)


    #Check for scaling factor
    def de_dt(s, e=None):
        '''
        Differential equation for eccentricity
        '''
        k = 304 / 15 * (s.G ** 3 * s.m_1 * s.m_2 * (s.m_1 + s.m_2) / s.c ** 5)
        if e == None:
            return -k * (1 / s.a_e(s.ecc) ** 4) * s.ecc / ((1 - s.ecc ** 2) ** (5 / 2)) * (1 + 121 / 304 * s.ecc ** 2)

        return -k * (1 / s.a_e(e) ** 4) * e / (1 - e ** 2) ** (5 / 2) * (1 + 121 / 304 * e ** 2)


    def set_a(s, a):
        '''
        Set a, the semi-major axis
        and update T  and a_til accordingly
        '''
        s.a = a
        s.T = np.sqrt((4 * np.pi ** 2 * s.m_red * s.a ** 3) / (s.G * s.m_1 * s.m_2))
        s.a_til = s.a / s.R_star
        return (s.a, s.a_til, s.T)


    def ecc_step(s, e_n, dt):
        '''
        Take a Runge-Kutta step on the eccentricity, and return e_next
        '''
        k_1 = dt * s.de_dt(e=e_n) #Euler step
        k_2 = dt * s.de_dt(e=e_n + 0.5 * k_1) #First correction
        k_3 = dt * s.de_dt(e=e_n + 0.5 * k_2) #Second correction
        k_4 = dt * s.de_dt(e=e_n + 0.5 * k_3) #Third correctionx

        return e_n + 1/6 * k_1 + 1 / 3 * k_2 + 1 / 3 * k_3 + 1 / 6 * k_4



    def last_orbits(s, ecc_start, t_f, num_points, a_min=0):
        '''
        simulate the last few orbits before the stars collide
        ecc_start: eccentricty to simulate the last few orbits from
        t_f: final time to end simulation at
        num_points: Number of points to include in simulation
        Returns: (t, ecc, a, T, th, h_p, h_c) at each point included in simulation
        '''
        print('Starting Last Orbits...')
        dt = t_f / num_points
        t_array = dt * np.arange(num_points)
        ecc_array = np.empty(t_array.size); ecc_array[0] = ecc_start; s.ecc = ecc_start

        a_array = np.empty(t_array.size)
        a_start = s.a_e(ecc_start); a_array[0] = a_start; s.a = a_start

        T_array = np.empty(t_array.size)
        T_start = np.sqrt((4 * np.pi ** 2 * s.m_red * a_start ** 3) / (s.G * s.m_1 * s.m_2))
        T_array[0] = T_start; s.T = T_start

        th_array = np.empty(t_array.size)
        th_start = s.th(t=0)
        th_array[0] = th_start

        h_p_array = np.empty(t_array.size)
        h_p_array[0] = s.h_plus(th_start)

        h_c_array = np.empty(t_array.size)
        h_c_array[0] = s.h_cross(th_start)

        for i in range(1, t_array.size):
            #Step eccentricity; then, update and calculate a, T, h_plus, h_cross
            #print(f'step: {i}')
            e_next = s.ecc_step(s.ecc, dt)

            if np.isnan(e_next) or s.a < a_min:
                print(f'Last step: {i}')
                t_array = t_array[:i-1]
                ecc_array = ecc_array[:i-1]
                a_array = a_array[:i-1]
                T_array = T_array[:i-1]
                th_array = th_array[:i-1]
                h_p_array = h_p_array[:i-1]
                h_c_array = h_c_array[:i-1]
                break

            #Store eccentricity history; update eccentricity
            ecc_array[i] = e_next
            s.ecc = e_next

            #Store semi-major axis history; update semi-major axis
            a_next = s.a_e(e_next)
            a_array[i] = a_next
            s.a = a_next

            #Store period history; update period
            #MAYBE THE 2PI I AM LOOKING FOR??
            T_next = np.sqrt((4 * np.pi ** 2 * s.m_red * a_next ** 3) / (s.G * s.m_1 * s.m_2))
            T_array[i] = T_next
            s.T = T_next

            th_next = s.th(t=t_array[i])
            th_array[i] = th_next

            h_p_array[i] = s.h_plus(th_next)
            h_c_array[i] = s.h_cross(th_next)

        print(f'Last orbits completed with parameters: ecc_0={ecc_start}, t_f={t_f}, n={num_points}, a_min={a_min}')
        return (t_array, ecc_array, a_array, T_array, th_array, h_p_array, h_c_array)

    def plot_last_orbits(s, ecc_start, t_f, num_points):
        def r(th_array, a_array, ecc_array):
            '''
            Return x, y position of star as a and ecc decay
            '''
            r = a_array * (1 - ecc_array) * (1 + ecc_array) / (1 + ecc_array * np.cos(th_array))
            return r

        t, ecc, a, T, th, h_p, h_c = s.last_orbits(ecc_start, t_f, num_points)


        a_1 = a / (1 + s.beta); a_2 = a - a_1
        r_decay = r(th,a_1,ecc)
        t = to_days(t)
        r_1 = to_au(np.array([r_decay * np.cos(th), r_decay * np.sin(th)]))
        r_2 = to_au(np.array([-r_decay * np.cos(th), -r_decay * np.sin(th)]))


        fig, axes = plt.subplots(2,figsize=(9,7))
        fig.suptitle(r'$\epsilon_i=$' + f'{ecc[0]}' + r' $m_1=$' + f'{round(to_sm(s.m_1),2)}' + r' $M_{\odot}$ ' + r'$m_2=$' + f'{round(to_sm(s.m_2),2)}' + r' $M_{\odot}$')
        #Frame about the larger orbit
        axes[0].set_xlim([1.1 * min(r_1[0].min(), r_2[0].min()), 1.1 * max(r_1[0].max(), r_2[0].max())])
        axes[0].set_ylim([1.1 * min(r_1[0].min(), r_2[0].min()), 1.1 * max(r_1[0].max(), r_2[0].max())])

        axes[1].set_xlim(t.min(), t.max())
        axes[1].set_ylim(min(1.1 * min(h_p.min(), h_c.min()), 0.9 * min(h_p.min(), h_c.min())),
                         max(1.1 * max(h_p.max(), h_c.max()), 0.9 * max(h_p.max(), h_c.max())))

        #Artist objects returned from plot function and to be used in animation
        axes[0].plot(r_1[0,:], r_1[1,:], color='pink')

        axes[0].plot(r_2[0,:], r_2[1,:], color='purple')

        axes[1].plot(t, h_p, color='green', label='h-plus')
        axes[1].plot(t, h_c, color='blue', label='h-cross')

        axes[0].legend(loc='upper right')
        axes[1].legend(loc='upper right')

        axes[0].set_xlabel('x (au)')
        axes[0].set_ylabel('y (au)')

        axes[1].set_xlabel(r'$t$ (s)')
        axes[1].set_ylabel('Wave Amplitude')

        print('Plotting Last Orbits')
        fig.show()

    def animate_last_orbits(s, ecc_start, t_f, num_points):
        def r(th_array, a_array, ecc_array):
            '''
            Return x, y position of star as a and ecc decay
            '''
            r = a_array * (1 - ecc_array) * (1 + ecc_array) / (1 + ecc_array * np.cos(th_array))
            return r

        def animate(f, t, r_1, r_2, h_p, h_c, orbit_1, orbit_2, line_1, line_2):
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

            line_1.set_data(t[:f], h_p[:f])
            line_2.set_data(t[:f], h_c[:f])

            return [orbit_1, orbit_2, line_1, line_2]

        t, ecc, a, T, th, h_p, h_c = s.last_orbits(ecc_start, t_f, num_points)
        t = t[0:-1:1]; ecc = ecc[0:-1:1]; a = a[0:-1:1]; T = T[0:-1:1]
        th = th[0:-1:1]; h_p = h_p[0:-1:1]; h_c = h_c[0:-1:1]


        a_1 = a / (1 + s.beta); a_2 = a - a_1
        r_decay = r(th,a_1,ecc)
        t = t / pc.c
        r_1 = to_au(np.array([r_decay * np.cos(th), r_decay * np.sin(th)]))
        r_2 = to_au(np.array([-r_decay * np.cos(th), -r_decay * np.sin(th)]))


        fig, axes = plt.subplots(2,figsize=(7,7))
        #Frame about the larger orbit
        axes[0].set_xlim([1.1 * min(r_1[0].min(), r_2[0].min()), 1.1 * max(r_1[0].max(), r_2[0].max())])
        axes[0].set_ylim([1.1 * min(r_1[0].min(), r_2[0].min()), 1.1 * max(r_1[0].max(), r_2[0].max())])

        axes[1].set_xlim(t.min(), t.max())
        axes[1].set_ylim(min(1.1 * min(h_p.min(), h_c.min()), 0.9 * min(h_p.min(), h_c.min())),
                         max(1.1 * max(h_p.max(), h_c.max()), 0.9 * max(h_p.max(), h_c.max())))

        #Artist objects returned from plot function and to be used in animation
        orbit_1, = axes[0].plot(r_1[0, :1], r_1[1, :1], 'o-', color='pink', \
                        label='Primary', mfc='r', mec='r', markersize=10, markevery=[-1])

        orbit_2, = axes[0].plot(r_2[0, :1], r_2[1, :1], 'o-', color='purple', \
                        label='Secondary', mfc='r', mec='r', markersize=10, markevery=[-1])

        line_1, = axes[1].plot(t[:1], h_p[:1], color='green', label='h-plus')
        line_2, = axes[1].plot(t[:1], h_c[:1], color='blue', label='h-cross')

        axes[0].legend(loc='upper right')
        axes[1].legend(loc='upper right')

        axes[0].set_xlabel('x (au)')
        axes[0].set_ylabel('y (au)')

        axes[1].set_xlabel(r'$t$ (days)')
        axes[1].set_ylabel('Wave Amplitude')

        #Choose interval to complete an orbit in 10s
        if (10 ** 3) / (num_points) > 0:
            interval = (10 ** 3) / (num_points)  #interval measured in milliseconds
        else:
            interval = 1 #interval cannot be less than 1

        ani = FuncAnimation(fig, animate, frames=th.size, interval=1, \
                            fargs=[t, r_1, r_2, h_p, h_c, orbit_1, orbit_2, line_1, line_2],repeat=True)

        plt.show()

if __name__ == '__main__':
    b = Binary_System(0.6, 1)
    b.animate_orbits(out_file='orbit.gif')
