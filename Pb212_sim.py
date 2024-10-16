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

def GaussCurve(x, a, mu, s):

    return a*np.exp(-(x-mu)**2/(2*(s**2)))

def main():

    # Define the detector resolution in MeV
    detres = 0.4
    detsig = detres/2.355

    # Define each alpha decay: energy and branching ratios
    # Normalize to 1
    xspace = np.linspace(4.5, 10, 1000)

    Bi212_alphaE = [6.05078, 6.08988, 5.768, 5.607]
    Bi212_alphaBr = [0.2513, 0.0975, 0.00611, 0.004061]
    Bi212_y0 = GaussCurve(xspace, Bi212_alphaBr[0], Bi212_alphaE[0], detsig)
    Bi212_y1 = GaussCurve(xspace, Bi212_alphaBr[1], Bi212_alphaE[1], detsig)
    Bi212_y2 = GaussCurve(xspace, Bi212_alphaBr[2], Bi212_alphaE[2], detsig)
    Bi212_y3 = GaussCurve(xspace, Bi212_alphaBr[3], Bi212_alphaE[3], detsig)
    Bi212_ysum = Bi212_y0 + Bi212_y1 + Bi212_y2 + Bi212_y3

    Po212_alphaE = [8.78486]
    Po212_alphaBr = [(1-np.sum(Bi212_alphaBr))] # alpha from beta branch
    Po212_y0 = GaussCurve(xspace, Po212_alphaBr[0], Po212_alphaE[0], detsig)
    Po212_ysum = Po212_y0



    # Summed spectrum
    # Shift up by 1
    ysum = Bi212_ysum + Po212_ysum + 1

    sumplt, sumax = plt.subplots(layout='constrained')
    sumax.plot(xspace, Bi212_ysum, linewidth=2.5, label='$^{212}$Bi')
    sumax.plot(xspace, Po212_ysum, linewidth=2.5, label='$^{212}$Po')
    sumax.plot(xspace, ysum, linewidth=2.5, linestyle='dashed', label='Total')
    sumax.set_xlabel('Energy (MeV)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.ylim(0,4)
   # plt.tight_layout()
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_Simulation.eps", format='eps')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_Simulation.png", format='png')
    sumplt.show()


main()