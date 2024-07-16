########################################################
# Created on June 27 2024
# @author: Johnathan Phillips
# @email: j.s.phillips@wustl.edu

# Purpose:

########################################################

# Import necessary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
import matplotlib as mpl
from pathlib import Path
from mpl_toolkits.mplot3d import axes3d

# Sets the plot style
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 100
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['xtick.minor.width'] = 1.0
plt.rcParams['xtick.major.size'] = 8.0
plt.rcParams['xtick.minor.size'] = 4.0
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.minor.bottom'] = True
plt.rcParams['xtick.minor.top'] = True
plt.rcParams['xtick.minor.visible'] = True


plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['ytick.minor.width'] = 1.0
plt.rcParams['ytick.major.size'] = 8.0
plt.rcParams['ytick.minor.size'] = 4.0
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.minor.right'] = True
plt.rcParams['ytick.minor.left'] = True
plt.rcParams['ytick.minor.visible'] = True

plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlepad'] = 15

def GaussFit(x, a, mu, s):

    return a * np.exp(-((x - mu)**2)/(2 * s**2))

def TripleGaussFit(x, a1, mu1, s1, a2, mu2, s2, a3, mu3, s3, bl):

    return GaussFit(x, a1, mu1, s1) + GaussFit(x, a2, mu2, s2) + GaussFit(x, a3, mu3, s3) + bl

def QuadGaussFit(x, a1, mu1, s1, a2, mu2, s2, a3, mu3, s3, a4, mu4, s4, bl):

    return GaussFit(x, a1, mu1, s1) + GaussFit(x, a2, mu2, s2) + GaussFit(x, a3, mu3, s3) + GaussFit(x, a4, mu4, s4) + bl


def main():

    # Make arrays containing the gaussian parameters for each isotope
    Ra_params = [1285.93446594, 1746.4214785, 183.95083003, 3850.22679643, 2157.57229319, 126.48844332, 2119.99706899, 2561.80926849, 130.85478446, 22.67620527]
    Ac_params = [2458.02500989, 1768.64425844, 91.44892971, 2672.62925512, 2018.58754993, 105.77633654, 2409.1151359, 2462.49872886, 110.42912563, 1705.32755114, 3295.84557428, 130.45108024, 26.49233856]

    xspace = np.linspace(0, 4095, 2000)

    indvplt, indvax = plt.subplots(layout='constrained')
    indvax.plot(xspace, TripleGaussFit(xspace, *Ra_params), linewidth=2.5, label=r'Ra-223')
    indvax.plot(xspace, QuadGaussFit(xspace, *Ac_params), linewidth=2.5, label=r'Ac-225')

    indvax.set_xlabel('ADC Channel')
    indvax.set_ylabel('Counts')
    indvplt.legend()
    indvplt.show()

    mixplt, mixvax = plt.subplots(layout='constrained')
    mixvax.plot(xspace, 0.01*TripleGaussFit(xspace, *Ra_params) + QuadGaussFit(xspace, *Ac_params), linewidth=2.5)

    mixvax.set_xlabel('ADC Channel')
    mixvax.set_ylabel('Counts')
    mixplt.show()

    Acplt, Acax = plt.subplots(layout='constrained')
    Acax.plot(xspace, 0.01 * TripleGaussFit(xspace, *Ra_params) + QuadGaussFit(xspace, *Ac_params), linewidth=2.5, label=r'Ac-225 + 1% Ra-223')
    Acax.plot(xspace, QuadGaussFit(xspace, *Ac_params), linewidth=2.5, label=r'Ac-225')

    #plt.ylim(0,4000)

    Acax.set_xlabel('ADC Channel')
    Acax.set_ylabel('Counts')
    Acax.legend('Upper left')
    Acplt.show()


main()