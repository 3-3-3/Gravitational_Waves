import numpy as np

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter

from Keppler_Equation.keppler import solve_keppler




class Binary_System:

    def __init__(s, ecc, T, R=100, m_1=10, m_2=6, th_p=0, i=0, th_n=0, phi=0, dec=0, ra=0, t=None):

        '''

        ecc: Eccentricity of orbit

        T: Orbital Period

	@@ -57,10 +57,12 @@ def __init__(s, ecc, T, R=100, m_1=10, m_2=1, th_p=0, i=0, th_n=0, phi=0, dec=0,

        s.e_cross = np.tensordot(s.a_hat, s.d_hat, 0) + np.tensordot(s.d_hat, s.a_hat, 0)




        if t == None:

            s.t = np.linspace(0, 2 * s.T, 2000)

        else:

            s.t = t




        plt.style.use('ggplot')







    def A_0(s, th):

        '''

	@@ -148,7 +150,7 @@ def r(th, a_p, ecc_p):




        return da_2




    def animate_orbits(s, out_file=None):

        '''

        Visualize star orbit (in the orbital plane)

        '''

	@@ -188,13 +190,13 @@ def animate(f, r_1, r_2, h_p, h_c, orbit_1, orbit_2, line_1, line_2):

        h_p = s.h_plus(th)

        h_c = s.h_cross(th)




        fig, axes = plt.subplots(2,figsize=(7,7))




        #Frame about the larger orbit

        axes[0].set_xlim([1.1 * min(r_1[0].min(), r_2[0].min()), 1.1 * max(r_1[0].max(), r_2[0].max())])

        axes[0].set_ylim([-1.1 * max(s.b_1, s.b_2), 1.1 * max(s.b_1, s.b_2)])




        axes[1].set_xlim(s.t.min(), s.t.max())

        axes[1].set_ylim(min(1.1 * min(h_p.min(), h_c.min()), 0.9 * min(h_p.min(), h_c.min())),

                         max(1.1 * max(h_p.max(), h_c.max()), 0.9 * max(h_p.max(), h_c.max())))




	@@ -211,18 +213,141 @@ def animate(f, r_1, r_2, h_p, h_c, orbit_1, orbit_2, line_1, line_2):

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




if __name__ == '__main__':

    b = Binary_System(0.6, 1)

    b.animate_orbits(out_file='orbit.gif')
