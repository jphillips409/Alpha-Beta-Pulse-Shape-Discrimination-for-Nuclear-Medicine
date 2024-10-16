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

def Calibration(x, a, b, c, d, e):

    return a * (x - d) + b * np.log(1 + c * (x - d)) + e

def main():

    # Define the detector resolution in MeV
    detres = 0.4
    detsig = detres/2.355

    # Define detector calibration, currently using a generalized one
    calparams = [1.14121109e-04, -2.30802018e+03, -8.62812074e-02, -4.33262770e-01, 1.61732905e-01]

    # Define each alpha decay: energy and branching ratios
    # Normalize to 1
    xspace = np.linspace(4, 9.5, 1000)

    Ac225_alphaE = [5.83, 5.7925, 5.7906, 5.732, 5.637, 5.724, 5.732, 5.682, 5.580, 5.609, 5.731]
    Ac225_alphaBr = [0.507, 0.181, 0.086, 0.08, 0.044, 0.031, 0.0132, 0.013, 0.012, 0.011, 0.0087]
    Ac225_y = [None] * len(Ac225_alphaE)
    Ac225_ysum = 0
    for i in range(len(Ac225_alphaE)):
        Ac225_y[i] = GaussCurve(xspace, Ac225_alphaBr[i], Ac225_alphaE[i], detsig)
        Ac225_ysum = Ac225_ysum + Ac225_y[i]

    Fr221_alphaE = [6.341, 6.1263, 6.2418]
    Fr221_alphaBr = [0.833, 0.1510, 0.0136]
    Fr221_y = [None] * len(Fr221_alphaE)
    Fr221_ysum = 0
    for i in range(len(Fr221_alphaE)):
        Fr221_y[i] = GaussCurve(xspace, Fr221_alphaBr[i], Fr221_alphaE[i], detsig)
        Fr221_ysum = Fr221_ysum + Fr221_y[i]

    At217_alphaE = [7.0669]
    At217_alphaBr = [0.9989]
    At217_y = [None] * len(At217_alphaE)
    At217_ysum = 0
    for i in range(len(At217_alphaE)):
        At217_y[i] = GaussCurve(xspace, At217_alphaBr[i], At217_alphaE[i], detsig)
        At217_ysum = At217_ysum + At217_y[i]

    Bi213_alphaE = [5.875, 5.558]
    Bi213_alphaBr = [0.01959, 0.00181]
    Bi213_y = [None] * len(Bi213_alphaE)
    Bi213_ysum = 0
    for i in range(len(Bi213_alphaE)):
        Bi213_y[i] = GaussCurve(xspace, Bi213_alphaBr[i], Bi213_alphaE[i], detsig)
        Bi213_ysum = Bi213_ysum + Bi213_y[i]

    Po213_alphaE = [8.376]
    Po213_alphaBr = [1 * 0.97860] # Beta branch
    Po213_y = [None] * len(Po213_alphaE)
    Po213_ysum = 0
    for i in range(len(Po213_alphaE)):
        Po213_y[i] = GaussCurve(xspace, Po213_alphaBr[i], Po213_alphaE[i], detsig)
        Po213_ysum = Po213_ysum + Po213_y[i]

    # Summed spectrum
    # Shift up by 1
    ysum = Ac225_ysum + Fr221_ysum + At217_ysum + Bi213_ysum + Po213_ysum + 1

    sumplt, sumax = plt.subplots(layout='constrained')
    sumax.plot(xspace, Ac225_ysum, linewidth=2.5, label='$^{225}$Ac')
    sumax.plot(xspace, Fr221_ysum, linewidth=2.5, label='$^{221}$Fr')
    sumax.plot(xspace, At217_ysum, linewidth=2.5, label='$^{217}$Ac')
    sumax.plot(xspace, Bi213_ysum, linewidth=2.5, label='$^{23}$Bi')
    sumax.plot(xspace, Po213_ysum, linewidth=2.5, label='$^{213}$Po')
    sumax.plot(xspace, ysum, linewidth=2.5, linestyle='dashed', label='Total')
    sumax.set_xlabel('Energy (MeV)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.ylim(0,4)
   # plt.tight_layout()
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ac225_Simulation.eps", format='eps')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ac225_Simulation.png", format='png')
    sumplt.show()

    # Calibrates the energy spectrum and replots it
    xspace_cal = Calibration(xspace, *calparams)

    sumplt_cal, sumax_cal = plt.subplots(layout='constrained')
    sumax_cal.plot(xspace_cal, Ac225_ysum, linewidth=2.5, label='$^{225}$Ac')
    sumax_cal.plot(xspace_cal, Fr221_ysum, linewidth=2.5, label='$^{221}$Fr')
    sumax_cal.plot(xspace_cal, At217_ysum, linewidth=2.5, label='$^{217}$Ac')
    sumax_cal.plot(xspace_cal, Bi213_ysum, linewidth=2.5, label='$^{23}$Bi')
    sumax_cal.plot(xspace_cal, Po213_ysum, linewidth=2.5, label='$^{213}$Po')
    sumax_cal.plot(xspace_cal, ysum, linewidth=2.5, linestyle='dashed', label='Total')
    sumax_cal.set_xlabel('Energy (MeV)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.ylim(0,4)
    #plt.xlim(0,4095)
    # plt.tight_layout()
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ac225_Simulation_cal.eps", format='eps')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ac225_Simulation_cal.png", format='png')
    sumplt_cal.show()

main()