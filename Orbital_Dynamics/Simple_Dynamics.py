#Adapted from ../mathematica_orbits.pdf, Orbital Dynamics with Mathematica
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.constants as pc

def r(t, r_p, ecc):
    '''
    t: Parameter t
    r_p: Pariapsis, or distance from focus to nearest point on ellipse, along the semi-major axis
    ecc: Eccentricity of ellipse, ranging from 0 for a circle to 1 for a parabola
    Returns: r to be used in parametric equation of ellipse
    '''
    r = r_p * (1 + ecc) / (1 + ecc * np.cos(t))
    return r

def animate(t, r_1, r_2, line_1, line_2):
    '''
    Frames being used as a proxy for parameter t
    r_1: List of position vectors at each time for orbit 1
    r_2: List of position vectors at each time for orbit 2
    line_1: artist for first orbit
    line_2: artist for second
    '''
    line_1.set_data(r_1[0, :t], r_1[1, :t])
    line_2.set_data(r_2[0, :t], r_2[1, :t])
    return [line_1, line_2]

if __name__ == '__main__':
    #Orbital parameters
    m_1 = 10; m_2 = 1 #Primary; secondary
    beta = m_2 / m_1 #mass ratio
    a_1 = 0.25 #Semi-major axis of first ellipse
    a_2 = a_1 / beta #Semi-major axis of second ellipse, determined by mass ratio and semi-major axis of first ellipse
    ecc = 0.5 #Eccentricity of orbits
    b_1 = np.sqrt((1 - ecc ** 2) * a_1 ** 2)
    b_2 = np.sqrt((1 - ecc ** 2) * a_2 ** 2)
    rp_1 = a_1 * (1 - ecc) #first pariapsis
    rp_2 = a_2 * (1 - ecc) #second pariapsis

    t = np.linspace(0, 2 * np.pi, 200)

    #Position vector for first orbit;
    r_1 = r(t, rp_1, ecc) * np.array([np.cos(t), np.sin(t)])
    #Postion vector for second orbit, parametrized so that
    #the two masses orbit the center of mass
    r_2 = r(t, rp_2, ecc) * np.array([-np.cos(t), -np.sin(t)])

    #Set up plot
    fig, ax = plt.subplots()
    ax.grid()

    #Frame about the larger orbit
    ax.set_xlim([-1.1 * max(a_1, a_2) * (1 - ecc), 1.1 * max(a_1, a_2) * (1 + ecc)])
    ax.set_ylim([-1.1 * max(b_1, b_2), 1.1 * max(b_1, b_2)])

    #Artist objects returned by plot (the first argument in a list, which we unpack) to be used in animation
    line_1, = ax.plot(r_1[0, :1], r_1[1, :1], 'o-', color='green', mfc='r', mec='r', markersize=10, markevery=[-1])
    line_2, = ax.plot(r_2[0, :1], r_2[1, :1], 'o-', color='blue', mfc='r', mec='r', markersize=10, markevery=[-1])

    #Animation: figure, animation function, number of frames to be iterated and passed to animate (first argument),
    #time interval between frames, in milliseconds, and list of additional arguments to be passed to the animate function
    ani = FuncAnimation(fig, animate, frames=t.size, interval=10, fargs=[r_1, r_2, line_1, line_2])
    plt.show()
