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

def MixedLnCurve(x, t1, t2):

    return LnCurve(x, t1) + LnCurve(x, t2)

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

    # Literature Pb212 half-life
    Pb212_t1half_lit = 10.628 # hours
    Pb212_halflives = (Pb212_days * 24)/Pb212_t1half_lit
    Pb212_halflives = np.array(Pb212_halflives)

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
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_DecayCurve.eps", format='eps')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_DecayCurve.png", format='png')
    fitpltPb.show()

    Pb212_err = np.sqrt(np.diag(fit_covPb))

    print("The Pb-212 half-life is ", 24*np.log(2)/np.abs(fit_paramPb[0]), " +/- ", np.abs((Pb212_err[0]/fit_paramPb[0])*24*np.log(2)/np.abs(fit_paramPb[0]))," hours")

    fit_paramPb_hl, fit_covPb_hl = curve_fit(LnCurve, xdata = Pb212_halflives, ydata = Pb212_LN, maxfev = 100000)
    print(fit_paramPb_hl)
    # Plot the Pb-212 as a function of half-lives
    xspace = np.linspace(0,Pb212_halflives[-1], 100)
    fitpltPb_hl, fitaxPb_hl = plt.subplots(layout = 'constrained')
    fitaxPb_hl.scatter(Pb212_halflives, Pb212_LN, linewidth = 2.5, label='$^{212}$Pb Data', color='black')
    fitaxPb_hl.plot(xspace,LnCurve(xspace,*fit_paramPb_hl), label='Linear Fit', color='red', linestyle='dashed')
    fitaxPb_hl.errorbar(Pb212_halflives, Pb212_LN, yerr=Pb212_LN_err, color='black', ls='none',  capsize=3, capthick=1, ecolor='black')
    fitaxPb_hl.set_xlabel('Half-Lives Since First Sample')
    fitaxPb_hl.set_ylabel('Ln(CPS/CPS$_{0}$)')
    fitaxPb_hl.legend(loc='upper right')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_DecayCurve_HalfLife.eps", format='eps')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_DecayCurve_HalfLife.png", format='png')
    fitpltPb_hl.show()


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
    fitaxAc.scatter(Ac225_days, Ac225_LN, linewidth = 2.5, label='$^{225}$Ac Data', color='black')
    fitaxAc.plot(xspace,LnCurve(xspace,*fit_paramAc), color='red', label='Decay Curve', linestyle='dashed')
    fitaxAc.set_xlabel('Time Since First Sample (Days)')
    fitaxAc.set_ylabel('Ln(CPS/CPS$_{0}$)')

    fitaxAc.legend(loc='upper right')
    fitpltAc.show()

    Ac225_err = np.sqrt(np.diag(fit_covAc))

    print("The Ac-225 half-life is ", np.log(2)/np.abs(fit_paramAc[0]), " +/- ", np.abs((Ac225_err[0]/fit_paramAc[0])*np.log(2)/np.abs(fit_paramAc[0]))," days")

    # Ra223 array for days since sample creation, CPS, and CPS uncertainty
    # Starts with first greased sample
    Ra223_days = [0, 0.899305556, 12.90833333, 15.22638889, 17.13819444, 35.94166667, 38.92986111, 39.97013889, 41.03819444, 43.13125, 46.11527778, 47.05902778]
    Ra223_days = np.array(Ra223_days)
    Ra223_CPS = [5330.683333, 5001.7, 2346.461111, 2020.6625, 1804.023333, 575.5833333, 477.8525, 449.2225, 419.6005556, 370.3883333, 312.8411111, 294.9183333]
    Ra223_CPS = np.array(Ra223_CPS)
    Ra223_CPSunc = [9.106728282, 9.030350061, 3.541242894, 2.865436994, 2.426687271, 0.691262432, 0.631039156, 0.017662391, 0.011380085, 0.010691931, 0.009826286, 0.009540659]
    Ra223_CPSunc = np.array(Ra223_CPSunc)

    Ra223_LN = np.log(Ra223_CPS/Ra223_CPS[0])
    Ra223_LN_err = np.abs((Ra223_CPSunc/Ra223_CPS)*Ra223_LN)

    paramRa = [-0.01]

    Ra223_thalf_lit = 11.4352 # days
    Ra223_thalf_lit_eq = -np.log(2)/Ra223_thalf_lit

    fit_paramRa, fit_covRa = curve_fit(LnCurve, xdata = Ra223_days, ydata = Ra223_LN, maxfev = 100000)

    xspace = np.linspace(0,Ra223_days[-1]+2, 100)

    fitpltRa, fitaxRa = plt.subplots(layout = 'constrained')
    fitaxRa.scatter(Ra223_days, Ra223_LN, linewidth = 2.5, label='$^{223}$Ra Data', color='black')
    fitaxRa.plot(xspace,LnCurve(xspace,*fit_paramRa), color='red', label='Decay Curve', linestyle='dashed')
    fitaxRa.plot(xspace,Ra223_thalf_lit_eq*xspace, color='blue', label='Literature Decay Curve', linestyle='dashed')
    fitaxRa.errorbar(Ra223_days, Ra223_LN, yerr=Ra223_LN_err, color='black', ls='none',  capsize=3, capthick=1, ecolor='black')
    fitaxRa.set_xlabel('Time Since First Sample (Days)')
    fitaxRa.set_ylabel('Ln(CPS/CPS$_{0}$)')

    fitpltRa.legend(loc='upper right')
    fitpltRa.show()

    Ra223_err = np.sqrt(np.diag(fit_covRa))

    print("The Ra-223 half-life is ", np.log(2)/np.abs(fit_paramRa[0]), " +/- ", np.abs((Ra223_err[0]/fit_paramRa[0])*np.log(2)/np.abs(fit_paramRa[0]))," days")

    # Ra223 array for days since sample creation, CPS, and CPS uncertainty
    # Starts with greased sample on 8/5/24
    Ra223_days = [0, 1.040277778, 2.108333333, 4.201388889, 7.185416667, 8.129166667]
    Ra223_days = np.array(Ra223_days)
    Ra223_CPS = [477.8525, 449.2225, 419.6005556, 370.3883333, 312.8411111, 294.9183333]
    Ra223_CPS = np.array(Ra223_CPS)
    Ra223_CPSunc = [0.631039156, 0.017662391, 0.011380085, 0.010691931, 0.009826286, 0.009540659]
    Ra223_CPSunc = np.array(Ra223_CPSunc)

    Ra223_LN = np.log(Ra223_CPS / Ra223_CPS[0])
    Ra223_LN_err = np.abs((Ra223_CPSunc / Ra223_CPS) * Ra223_LN)

    paramRa = [-0.01]

    Ra223_thalf_lit = 11.4352  # days
    Ra223_thalf_lit_eq = -np.log(2) / Ra223_thalf_lit

    fit_paramRa, fit_covRa = curve_fit(LnCurve, xdata=Ra223_days, ydata=Ra223_LN, maxfev=100000)

    xspace = np.linspace(0, Ra223_days[-1] + 2, 100)

    fitpltRa, fitaxRa = plt.subplots(layout='constrained')
    fitaxRa.scatter(Ra223_days, Ra223_LN, linewidth=2.5, label='$^{223}$Ra Data', color='black')
    fitaxRa.plot(xspace, LnCurve(xspace, *fit_paramRa), color='red', label='Decay Curve', linestyle='dashed')
    fitaxRa.plot(xspace, Ra223_thalf_lit_eq * xspace, color='blue', label='Literature Decay Curve', linestyle='dashed')
    fitaxRa.errorbar(Ra223_days, Ra223_LN, yerr=Ra223_LN_err, color='black', ls='none', capsize=3, capthick=1,
                     ecolor='black')
    fitaxRa.set_xlabel('Time Since First Sample (Days)')
    fitaxRa.set_ylabel('Ln(CPS/CPS$_{0}$)')

    fitpltRa.legend(loc='upper right')
    fitpltRa.show()

    Ra223_err = np.sqrt(np.diag(fit_covRa))

    print("The Ra-223 half-life is ", np.log(2) / np.abs(fit_paramRa[0]), " +/- ",
          np.abs((Ra223_err[0] / fit_paramRa[0]) * np.log(2) / np.abs(fit_paramRa[0])), " days")

    # Fits Ra-223 with two mixed decay curves
    # fit_paramRa_mixed, fit_covRa_mixed = curve_fit(MixedLnCurve, xdata = Ra223_days, ydata = Ra223_LN, maxfev = 100000, bounds=((8, -np.inf, 6, -np.inf),(np.inf,0, np.inf, 0)))
    # print(fit_paramRa_mixed)
    # fitpltRa_mixed, fitaxRa_mixed = plt.subplots(layout = 'constrained')
    # fitaxRa_mixed.scatter(Ra223_days, Ra223_LN, linewidth = 2.5, label='$^{223}$Ra Data', color='black')
    # fitaxRa_mixed.plot(xspace,MixedLnCurve(xspace,*fit_paramRa_mixed), color='red', label='Decay Curve', linestyle='dashed')
    # fitaxRa_mixed.plot(xspace,LnCurve(xspace,fit_paramRa_mixed[0],fit_paramRa_mixed[1]), color='blue', label='t1', linestyle='dashed')
    # fitaxRa_mixed.plot(xspace,LnCurve(xspace,fit_paramRa_mixed[2],fit_paramRa_mixed[3]), color='green', label='t2', linestyle='dashed')
    # fitaxRa_mixed.set_xlabel('Time Since First Sample (Days)')
    # fitaxRa_mixed.set_ylabel('Ln(CPS/CPS$_{0}$)')
    #
    # fitpltRa_mixed.legend(loc='upper right')
    # fitpltRa_mixed.show()

main()