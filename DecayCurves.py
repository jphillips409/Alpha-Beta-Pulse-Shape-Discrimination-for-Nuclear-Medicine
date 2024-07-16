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

def LinCurve(x, a, t):

    return a * np.exp(-t*x)

def LnCurve(x, t):

    return t*x

def main():

    # Pb212 array for days since sample creation
    Pb212_days = [0, 0.800694444, 1.089583333, 1.768055556, 2.079166667, 2.797222222, 3.093055556, 3.83125, 4.952777778, 5.8875, 6.90625]
    Pb212_days = np.array(Pb212_days)
    Pb212_CPS = [1295.116667, 389.5533333, 246.2275, 83.66708333, 51.2125, 16.41555556, 10.49407407, 3.169305556, 0.522916667, 0.119097222, 0.025555556]
    Pb212_CPS = np.array(Pb212_CPS)
    Pb212_CPSunc = [4.645995887, 0.805763958, 0.452978568, 0.186711769, 0.119271516, 0.067526858, 0.062343357, 0.020980499, 0.006026081, 0.002875872, 0.001332175]
    Pb212_CPSunc = np.array(Pb212_CPSunc)

    Pb212_LN = np.log(Pb212_CPS/Pb212_CPS[0])
    Pb212_LN = np.array(Pb212_LN)
    Pb212_LN_err = np.abs((Pb212_CPSunc/Pb212_CPS)*Pb212_LN)

    paramPb = [-0.01]

    fit_paramPb, fit_covPb = curve_fit(LnCurve, xdata = Pb212_days, ydata = Pb212_LN, maxfev = 100000)

    xspace = np.linspace(0,6.90625, 100)

    fitpltPb, fitaxPb = plt.subplots(layout = 'constrained')
    fitaxPb.scatter(Pb212_days, Pb212_LN, linewidth = 2.5, label='$^{212}$Pb Data', color='black')
    fitaxPb.plot(xspace,LnCurve(xspace,*fit_paramPb), label='Linear Fit', color='red', linestyle='dashed')
    fitaxPb.errorbar(Pb212_days, Pb212_LN, yerr=Pb212_LN_err, color='black', ls='none',  capsize=3, capthick=1, ecolor='black')
    fitaxPb.set_xlabel('Time Since First Sample (Days)')
    fitaxPb.set_ylabel('Ln(CPS/CPS$_{0}$)')

    fitaxPb.legend(loc='upper right')
    fitpltPb.show()

    print("The Pb-212 half-life is ", 24*np.log(2)/np.abs(fit_paramPb[0]), " hours")


    # Ac225 array for days since sample creation, CPS, and CPS uncertainty
    Ac225_days = [0, 1.01875, 3.879166667, 5.104166667]
    Ac225_days = np.array(Ac225_days)
    Ac225_CPS = [11352.13333, 10572.16667, 8642.416667, 7964.216667]
    Ac225_CPS = np.array(Ac225_CPS)
    Ac225_CPSunc = [13.75507987, 13.27413944, 12.00167812, 11.52115205]
    Ac225_CPSunc = np.array(Ac225_CPSunc)

    Ac225_LN = np.log(Ac225_CPS/11352.13333)

    paramAc = [-0.01]

    fit_paramAc, fit_covAc = curve_fit(LnCurve, xdata = Ac225_days, ydata = Ac225_LN, maxfev = 100000)

    xspace = np.linspace(0,6, 100)

    fitpltAc, fitaxAc = plt.subplots(layout = 'constrained')
    fitaxAc.scatter(Ac225_days, Ac225_LN, linewidth = 2.5, label='$^{225}$Ac Data', color='blue')
    fitaxAc.plot(xspace,LnCurve(xspace,*fit_paramAc), color='blue', label='Decay Curve', linestyle='dashed')
    fitaxAc.set_xlabel('Time Since First Sample (Days)')
    fitaxAc.set_ylabel('Ln(CPS/CPS$_{0}$)')

    fitaxAc.legend(loc='upper right')
    fitpltAc.show()

    print("The Ac-225 half-life is ", np.log(2)/np.abs(fit_paramAc[0]), " days")

main()