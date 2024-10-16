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
    detres = 0.5
    detsig = detres/2.355

    # Define detector calibration, currently using Ra223 and Ac225 from thin grease
    calparams = [1.92518419e-03, -3.05271172e+03, -7.96657258e-02, 7.12182112e-01, 9.54110159e-01]

    # Define each alpha decay: energy and branching ratios
    # Normalize to 1
    xspace = np.linspace(4.5, 8.5, 1000)

    Ac227_alphaE = [4.95326, 4.9407, 4.8727, 4.855, 4.768, 4.796]
    Ac227_alphaBr = [0.00658, 0.00546, 0.00087, 0.00083, 0.00025, 0.00014]
    Ac227_y0 = GaussCurve(xspace, Ac227_alphaBr[0], Ac227_alphaE[0], detsig)
    Ac227_y1 = GaussCurve(xspace, Ac227_alphaBr[1], Ac227_alphaE[1], detsig)
    Ac227_y2 = GaussCurve(xspace, Ac227_alphaBr[2], Ac227_alphaE[2], detsig)
    Ac227_y3 = GaussCurve(xspace, Ac227_alphaBr[3], Ac227_alphaE[3], detsig)
    Ac227_y4 = GaussCurve(xspace, Ac227_alphaBr[4], Ac227_alphaE[4], detsig)
    Ac227_y5 = GaussCurve(xspace, Ac227_alphaBr[5], Ac227_alphaE[5], detsig)
    Ac227_ysum = Ac227_y0 + Ac227_y1 + Ac227_y2 + Ac227_y3 + Ac227_y4 + Ac227_y5

    Th227_alphaE = [6.03801, 5.97772, 5.75687, 5.7088, 5.7132, 5.7008, 5.9597, 6.0088, 5.8666, 5.668, 5.693, 5.8075, 5.916, 5.7955]
    Th227_alphaBr = [0.242, 0.235, 0.204, 0.083, 0.0489, 0.0363, 0.03, 0.029, 0.0242, 0.0206, 0.015, 0.0127, 0.0078, 0.00311]
    Th227_y0 = GaussCurve(xspace, Th227_alphaBr[0], Th227_alphaE[0], detsig)
    Th227_y1 = GaussCurve(xspace, Th227_alphaBr[1], Th227_alphaE[1], detsig)
    Th227_y2 = GaussCurve(xspace, Th227_alphaBr[2], Th227_alphaE[2], detsig)
    Th227_y3 = GaussCurve(xspace, Th227_alphaBr[3], Th227_alphaE[3], detsig)
    Th227_y4 = GaussCurve(xspace, Th227_alphaBr[4], Th227_alphaE[4], detsig)
    Th227_y5 = GaussCurve(xspace, Th227_alphaBr[5], Th227_alphaE[5], detsig)
    Th227_y6 = GaussCurve(xspace, Th227_alphaBr[6], Th227_alphaE[6], detsig)
    Th227_y7 = GaussCurve(xspace, Th227_alphaBr[7], Th227_alphaE[7], detsig)
    Th227_y8 = GaussCurve(xspace, Th227_alphaBr[8], Th227_alphaE[8], detsig)
    Th227_y9 = GaussCurve(xspace, Th227_alphaBr[9], Th227_alphaE[9], detsig)
    Th227_y10 = GaussCurve(xspace, Th227_alphaBr[10], Th227_alphaE[10], detsig)
    Th227_y11 = GaussCurve(xspace, Th227_alphaBr[11], Th227_alphaE[11], detsig)
    Th227_y12 = GaussCurve(xspace, Th227_alphaBr[12], Th227_alphaE[12], detsig)
    Th227_y13 = GaussCurve(xspace, Th227_alphaBr[13], Th227_alphaE[13], detsig)
    Th227_ysum = Th227_y0 + Th227_y1 + Th227_y2 + Th227_y3 + Th227_y4 + Th227_y5 + Th227_y6 + Th227_y7 + Th227_y8 + Th227_y9 + Th227_y10 + Th227_y11 + Th227_y12 + Th227_y13

    Ra223_alphaE = [5.7162, 5.6067, 5.5398, 5.747, 5.4336, 5.5016, 5.8713]
    Ra223_alphaBr = [0.512, 0.25, 0.089, 0.089, 0.018, 0.01, 0.01]
    Ra223_y0 = GaussCurve(xspace, Ra223_alphaBr[0], Ra223_alphaE[0], detsig)
    Ra223_y1 = GaussCurve(xspace, Ra223_alphaBr[1], Ra223_alphaE[1], detsig)
    Ra223_y2 = GaussCurve(xspace, Ra223_alphaBr[2], Ra223_alphaE[2], detsig)
    Ra223_y3 = GaussCurve(xspace, Ra223_alphaBr[3], Ra223_alphaE[3], detsig)
    Ra223_y4 = GaussCurve(xspace, Ra223_alphaBr[4], Ra223_alphaE[4], detsig)
    Ra223_y5 = GaussCurve(xspace, Ra223_alphaBr[5], Ra223_alphaE[5], detsig)
    Ra223_y6 = GaussCurve(xspace, Ra223_alphaBr[6], Ra223_alphaE[6], detsig)
    Ra223_ysum = Ra223_y0 + Ra223_y1 + Ra223_y2 + Ra223_y3 + Ra223_y4 + Ra223_y5 + Ra223_y6

    Rn219_alphaE = [6.8191, 6.5526, 6.425]
    Rn219_alphaBr = [0.794, 0.129, 0.075]
    Rn219_y0 = GaussCurve(xspace, Rn219_alphaBr[0], Rn219_alphaE[0], detsig)
    Rn219_y1 = GaussCurve(xspace, Rn219_alphaBr[1], Rn219_alphaE[1], detsig)
    Rn219_y2 = GaussCurve(xspace, Rn219_alphaBr[2], Rn219_alphaE[2], detsig)
    Rn219_ysum = Rn219_y0 + Rn219_y1 + Rn219_y2

    Po215_alphaE = [7.3861]
    Po215_alphaBr = [0.9999977]
    Po215_y0 = GaussCurve(xspace, Po215_alphaBr[0], Po215_alphaE[0], detsig)

    Bi211_alphaE = [6.6229, 6.2782]
    Bi211_alphaBr = [0.8354, 0.1619]
    Bi211_y0 = GaussCurve(xspace, Bi211_alphaBr[0], Bi211_alphaE[0], detsig)
    Bi211_y1 = GaussCurve(xspace, Bi211_alphaBr[1], Bi211_alphaE[1], detsig)
    Bi211_ysum = Bi211_y0 + Bi211_y1

    # Summed spectrum
    # Shift up by 1
    # Includes Ac-227 and Th-227
    ysum_AcTh = Ac227_ysum + Th227_ysum + Ra223_ysum + Rn219_ysum + Po215_y0 + Bi211_ysum + 1

    # Includes Th-227
    ysum_Th = Th227_ysum + Ra223_ysum + Rn219_ysum + Po215_y0 + Bi211_ysum + 1

    # no Ac-227 or Th-227
    ysum = Ra223_ysum + Rn219_ysum + Po215_y0 + Bi211_ysum + 1

    AcThsumplt, AcThsumax = plt.subplots(layout='constrained')
    AcThsumax.plot(xspace, Ac227_ysum, linewidth=2.5, label='$^{227}$Ac')
    AcThsumax.plot(xspace, Th227_ysum, linewidth=2.5, label='$^{227}$Th')
    AcThsumax.plot(xspace, Ra223_ysum, linewidth=2.5, label='$^{223}$Ra')
    AcThsumax.plot(xspace, Rn219_ysum, linewidth=2.5, label='$^{219}$Rn')
    AcThsumax.plot(xspace, Po215_y0, linewidth=2.5, label='$^{215}$Po')
    AcThsumax.plot(xspace, Bi211_ysum, linewidth=2.5, label='$^{211}$Bi')
    AcThsumax.plot(xspace, ysum_AcTh, linewidth=2.5, linestyle='dashed', label='Total')
    AcThsumax.set_xlabel('Energy (MeV)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    AcThsumplt.show()

    Thsumplt, Thsumax = plt.subplots(layout='constrained')
    Thsumax.plot(xspace, Th227_ysum, linewidth=2.5, label='$^{227}$Th')
    Thsumax.plot(xspace, Ra223_ysum, linewidth=2.5, label='$^{223}$Ra')
    Thsumax.plot(xspace, Rn219_ysum, linewidth=2.5, label='$^{219}$Rn')
    Thsumax.plot(xspace, Po215_y0, linewidth=2.5, label='$^{215}$Po')
    Thsumax.plot(xspace, Bi211_ysum, linewidth=2.5, label='$^{211}$Bi')
    Thsumax.plot(xspace, ysum_Th, linewidth=2.5, linestyle='dashed', label='Total')
    Thsumax.set_xlabel('Energy (MeV)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    Thsumplt.show()

    sumplt, sumax = plt.subplots(layout='constrained')
    sumax.plot(xspace, Ra223_ysum, linewidth=2.5, label='$^{223}$Ra')
    sumax.plot(xspace, Rn219_ysum, linewidth=2.5, label='$^{219}$Rn')
    sumax.plot(xspace, Po215_y0, linewidth=2.5, label='$^{215}$Po')
    sumax.plot(xspace, Bi211_ysum, linewidth=2.5, label='$^{211}$Bi')
    sumax.plot(xspace, ysum, linewidth=2.5, linestyle='dashed', label='Total')
    sumax.set_xlabel('Energy (MeV)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.ylim(0,4)
   # plt.tight_layout()
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ra223_Simulation.eps", format='eps')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ra223_Simulation.png", format='png')
    sumplt.show()

    # Calibrates the energy spectrum and replots it
    xspace_cal = Calibration(xspace, *calparams)

    sumplt_cal, sumax_cal = plt.subplots(layout='constrained')
    sumax_cal.plot(xspace_cal, Ra223_ysum, linewidth=2.5, label='$^{223}$Ra')
    sumax_cal.plot(xspace_cal, Rn219_ysum, linewidth=2.5, label='$^{219}$Rn')
    sumax_cal.plot(xspace_cal, Po215_y0, linewidth=2.5, label='$^{215}$Po')
    sumax_cal.plot(xspace_cal, Bi211_ysum, linewidth=2.5, label='$^{211}$Bi')
    sumax_cal.plot(xspace_cal, ysum, linewidth=2.5, linestyle='dashed', label='Total')
    sumax_cal.set_xlabel('Energy (MeV)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.ylim(0,4)
    #plt.xlim(0,4095)
    # plt.tight_layout()
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ra223_Simulation_cal.eps", format='eps')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ra223_Simulation_cal.png", format='png')
    sumplt_cal.show()


main()