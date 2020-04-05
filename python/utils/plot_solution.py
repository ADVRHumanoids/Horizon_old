import matplotlib.pyplot as plt
from utils.conversions_to_euler import *

def plot_solution(q):

    plt.subplot(211)
    plt.plot(q[:, 0:3])
    plt.xlabel('$\mathrm{node}$')
    plt.ylabel('$\mathrm{Position}$')
    plt.grid()
    plt.subplot(212)
    plt.plot(quaternion_to_euler(q[:, 3:7]))
    plt.xlabel('$\mathrm{node}$')
    plt.ylabel('$\mathrm{Orientation} \quad [XYZ]$')
    plt.grid()
    plt.suptitle('$\mathrm{Floating Base}$')
    plt.show()
    plt.savefig('line_plot.pdf')  
