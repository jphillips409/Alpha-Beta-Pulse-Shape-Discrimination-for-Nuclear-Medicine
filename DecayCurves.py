########################################################
# Created on June 27 2024
# @author: Johnathan Phillips
# @email: j.s.phillips@wustl.edu

# Purpose:

########################################################

# Import necessary modules
import numpy as np
from matplotlib.pylab import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.cm as cm
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

def MixedLnLinCurve(x, t1, t2,a):

    return LnCurve(x, t1) + LnCurve(x, t2) + a

def Activity(x,A,t):

    return A*np.exp(-t*x)

def PbActivity(x,A):

    return A*np.exp(-(np.log(2)/10.628)*x)

def BiActivity(x,A):

    return A*np.exp(-(np.log(2)/(60.551/60))*x)

def BiGrowth(x,A1):

    t1 = np.log(2)/10.628 # Pb-212 decay constant
    t2 = np.log(2)/(60.551/60) # Bi-212 decay constant
    return (t2/(t2-t1))*A1*(np.exp(-t1*x)-np.exp(-t2*x))

# For transient equilibrium, t1 and t2 are not parameters
def Bi212TransEq(x,A1,A2):
    t1 = np.log(2)/10.628 # Pb-212 decay constant
    t2 = np.log(2)/(60.551/60) # Bi-212 decay constant
    return A1*np.exp(-t1*x) + (t2/(t2-t1))*A1*(np.exp(-t1*x)-np.exp(-t2*x)) + A2*np.exp(-t2*x)

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
    min_val = 0
    max_val = 100000
    my_cmap = cm.get_cmap('jet')  # or any other one
    norm = matplotlib.colors.Normalize(min_val, max_val)  # the color maps work for [0, 1]

    cmmapable = cm.ScalarMappable(norm, my_cmap)
    cmmapable.set_array(range(min_val, max_val))

    fitpltPb, fitaxPb = plt.subplots(layout = 'constrained')
    fitaxPb.scatter(Pb212_days, Pb212_LN, linewidth = 2.5, label='$^{212}$Pb Data', color='black')
    fitaxPb.plot(xspace,LnCurve(xspace,*fit_paramPb), label='Linear Fit', color='red', linestyle='dashed')
    fitaxPb.errorbar(Pb212_days, Pb212_LN, yerr=Pb212_LN_err, color='black', ls='none',  capsize=3, capthick=1, ecolor='black')
    fitaxPb.set_xlabel('Time Since First Sample (Days)',fontsize=20)
    fitaxPb.set_ylabel('Ln(CPS/CPS$_{0}$)',fontsize=20)
    #cbar = plt.colorbar(cmmapable, ax=fitaxPb, alpha=0)

    # remove the tick labels
    #cbar.ax.set_yticklabels(['', ''])
    # set the tick length to 0
    #cbar.ax.tick_params(axis='y', which="both", length=0)
    # set everything that has a linewidth to 0
    #for a in cbar.ax.get_children():
        #try:
           # a.set_linewidth(0)
       # except:
           # pass
    fitaxPb.legend(loc='upper right',fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_DecayCurve.eps", format='eps')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_DecayCurve.png", format='png')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_DecayCurve.svg", format='svg')

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



    # Pb212 in regular UG and cleared blood
    # Pb212 array for days since sample creation
    Pb212_CBlood_days = [0,0.705555556,1.886805556,2.681944444,3.585416667,4.567361111,5.679861111,6.567361111,7.634722222]
    Pb212_CBlood_days = np.array(Pb212_CBlood_days)
    Pb212_CBlood_CPS = [1391.991667,497.99375,78.58583333,20.91675926,4.829444444,0.99037037,0.164756944,0.045,0.013645833]
    Pb212_CBlood_CPS = np.array(Pb212_CBlood_CPS)
    Pb212_CBlood_CPSunc = [2.408311707,1.018571048,0.255906613,0.044008374,0.016379941,0.006771298,0.002391804,0.00125,0.000688341]
    Pb212_CBlood_CPSunc = np.array(Pb212_CBlood_CPSunc)

    Pb212_CBlood_LN = np.log(Pb212_CBlood_CPS / Pb212_CBlood_CPS[0])
    Pb212_CBlood_LN = np.array(Pb212_CBlood_LN)
    Pb212_CBlood_LN_err = np.abs((Pb212_CBlood_CPSunc / Pb212_CBlood_CPS) * Pb212_CBlood_LN)

    # Literature Pb212 half-life
    Pb212_CBlood_t1half_lit = 10.628  # hours
    Pb212_CBlood_halflives = (Pb212_CBlood_days * 24) / Pb212_CBlood_t1half_lit
    Pb212_CBlood_halflives = np.array(Pb212_CBlood_halflives)

    paramPb = [-0.01]

    fit_paramPb_CBlood, fit_covPb_CBlood = curve_fit(LnCurve, xdata=Pb212_CBlood_days, ydata=Pb212_CBlood_LN, maxfev=100000)

    xspace = np.linspace(0,  Pb212_CBlood_days[-1], 100)

    fitpltPb_CBlood, fitaxPb_CBlood = plt.subplots(layout='constrained')
    fitaxPb_CBlood.scatter(Pb212_CBlood_days, Pb212_CBlood_LN, linewidth=2.5, label='$^{212}$Pb Data', color='black')
    fitaxPb_CBlood.plot(xspace, LnCurve(xspace, *fit_paramPb_CBlood), label='Linear Fit', color='red', linestyle='dashed')
    fitaxPb_CBlood.errorbar(Pb212_CBlood_days, Pb212_CBlood_LN, yerr=Pb212_CBlood_LN_err, color='black', ls='none', capsize=3, capthick=1,
                     ecolor='black')
    fitaxPb_CBlood.set_xlabel('Time Since First Sample (Days)')
    fitaxPb_CBlood.set_ylabel('Ln(CPS/CPS$_{0}$)')

    fitaxPb_CBlood.legend(loc='upper right')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_CBlood_DecayCurve.eps", format='eps')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_CBlood_DecayCurve.png", format='png')
    fitpltPb_CBlood.show()

    Pb212_CBlood_err = np.sqrt(np.diag(fit_covPb_CBlood))

    print("The Pb-212 half-life is ", 24 * np.log(2) / np.abs(fit_paramPb_CBlood[0]), " +/- ",
          np.abs((Pb212_CBlood_err[0] / fit_paramPb_CBlood[0]) * 24 * np.log(2) / np.abs(fit_paramPb_CBlood[0])), " hours")

    fit_paramPb_CBlood_hl, fit_covPb_CBlood_hl = curve_fit(LnCurve, xdata=Pb212_CBlood_halflives, ydata=Pb212_CBlood_LN, maxfev=100000)
    print(fit_paramPb_CBlood_hl)
    # Plot the Pb-212 as a function of half-lives
    xspace = np.linspace(0, Pb212_CBlood_halflives[-1], 100)
    fitpltPb_CBlood_hl, fitaxPb_CBlood_hl = plt.subplots(layout='constrained')
    fitaxPb_CBlood_hl.scatter(Pb212_CBlood_halflives, Pb212_CBlood_LN, linewidth=2.5, label='$^{212}$Pb Data', color='black')
    fitaxPb_CBlood_hl.plot(xspace, LnCurve(xspace, *fit_paramPb_CBlood_hl), label='Linear Fit', color='red', linestyle='dashed')
    fitaxPb_CBlood_hl.errorbar(Pb212_CBlood_halflives, Pb212_CBlood_LN, yerr=Pb212_CBlood_LN_err, color='black', ls='none', capsize=3, capthick=1,
                        ecolor='black')
    fitaxPb_CBlood_hl.set_xlabel('Half-Lives Since First Sample')
    fitaxPb_CBlood_hl.set_ylabel('Ln(CPS/CPS$_{0}$)')
    fitaxPb_CBlood_hl.legend(loc='upper right')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_CBlood_DecayCurve_HalfLife.eps",
                format='eps')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_CBlood_DecayCurve_HalfLife.png",
                format='png')
    fitpltPb_CBlood_hl.show()

    # Pb212 Gamma Counter Data
    # Pb212 array for days since sample creation
    Pb212_GC_days = [0.00,0.82,1.04,1.71,1.92,2.87,3.83,4.74,5.73,6.72]
    Pb212_GC_days = np.array(Pb212_GC_days)

    Pb212_GC_LN = [0,-1.2698493,-1.608356,-2.6697923,-2.9876657,-4.4026968,-5.6988116,-6.5011448,-6.91355,-6.9635311]
    Pb212_GC_LN = np.array(Pb212_GC_LN)
    Pb212_GC_LN_err = [0,0.012952463,0.019461108,0.054730742,0.071703977,0.215291874,0.537397934,0.930963935,1.216093445,1.296609491]

    #for fitting
    Pb212_GC_days_fit = [0.00,0.82,1.04,1.71,1.92,2.87,3.83,4.74,5.73,6.72]
    Pb212_GC_days_fit = np.array(Pb212_GC_days_fit)
    Pb212_GC_LN_fit = [0,-1.2698493,-1.608356,-2.6697923,-2.9876657,-4.4026968,-5.6988116,-6.5011448,-6.91355,-6.9635311]
    Pb212_GC_LN_fit = np.array(Pb212_GC_LN_fit)
    Pb212_GC_LN_err_fit = [0,-0.012952463,-0.019461108,-0.054730742,-0.071703977,-0.215291874,-0.537397934,-0.930963935,-1.216093445,-1.296609491]
    # Literature Pb212 half-life
    Pb212_GC_t1half_lit = 10.628  # hours
    Pb212_GC_halflives = (Pb212_GC_days * 24) / Pb212_GC_t1half_lit
    Pb212_GC_halflives = np.array(Pb212_GC_halflives)

    Pb212_GC_halflives_fit = (Pb212_GC_days_fit * 24) / Pb212_GC_t1half_lit
    Pb212_GC_halflives_fit = np.array(Pb212_GC_halflives_fit)

    paramPb = [-0.01]

    fit_paramPb_GC, fit_covPb_GC = curve_fit(LnCurve, xdata=Pb212_GC_days_fit, ydata=Pb212_GC_LN_fit, maxfev=100000)

    xspace = np.linspace(0,  Pb212_GC_days[-1], 100)

    fitpltPb_GC, fitaxPb_GC = plt.subplots(layout='constrained')
    fitaxPb_GC.scatter(Pb212_GC_days, Pb212_GC_LN, linewidth=2.5, label='$^{212}$Pb Data', color='black')
    fitaxPb_GC.plot(xspace, LnCurve(xspace, *fit_paramPb_GC), label='Linear Fit', color='red', linestyle='dashed')
    fitaxPb_GC.errorbar(Pb212_GC_days, Pb212_GC_LN, yerr=Pb212_GC_LN_err, color='black', ls='none', capsize=3, capthick=1,
                     ecolor='black')
    fitaxPb_GC.set_xlabel('Time Since First Sample (Days)')
    fitaxPb_GC.set_ylabel('Ln(CPS/CPS$_{0}$)')

    fitaxPb_GC.legend(loc='upper right')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_GC_DecayCurve.eps", format='eps')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_GC_DecayCurve.png", format='png')
    fitpltPb_GC.show()

    Pb212_GC_err = np.sqrt(np.diag(fit_covPb_GC))
    print("The Pb-212 half-life from the GC is  ", 24*np.log(2)/np.abs(fit_paramPb_GC[0]), " +/- ", np.abs((Pb212_GC_err[0]/fit_paramPb_GC[0])*24*np.log(2)/np.abs(fit_paramPb_GC[0]))," hours")

    fit_paramPb_GC_hl, fit_covPb_GC_hl = curve_fit(LnCurve, xdata=Pb212_GC_halflives_fit, ydata=Pb212_GC_LN_fit, maxfev=100000)
    print(fit_paramPb_CBlood_hl)
    # Plot the Pb-212 as a function of half-lives
    xspace = np.linspace(0, Pb212_GC_halflives[-1], 100)
    fitpltPb_GC_hl, fitaxPb_GC_hl = plt.subplots(layout='constrained')
    fitaxPb_GC_hl.scatter(Pb212_GC_halflives, Pb212_GC_LN, linewidth=2.5, label='$^{212}$Pb Data', color='black')
    fitaxPb_GC_hl.plot(xspace, LnCurve(xspace, *fit_paramPb_GC_hl), label='Linear Fit', color='red', linestyle='dashed')
    fitaxPb_GC_hl.errorbar(Pb212_GC_halflives, Pb212_GC_LN, yerr=Pb212_GC_LN_err, color='black', ls='none', capsize=3, capthick=1,
                        ecolor='black')
    fitaxPb_GC_hl.set_xlabel('Half-Lives Since First Sample')
    fitaxPb_GC_hl.set_ylabel('Ln(CPS/CPS$_{0}$)')
    fitaxPb_GC_hl.legend(loc='upper right')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_GC_DecayCurve_HalfLife.eps",
                format='eps')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_GC_DecayCurve_HalfLife.png",
                format='png')
    fitpltPb_GC_hl.show()

    # Pb212 GC Data
    # Pb212 array for days since sample creation
    Pb212_HPGe_days = [0.00,0.78,1.02,1.75,2.04,2.77,3.88,4.73,5.78,6.72,7.78]
    Pb212_HPGe_days = np.array(Pb212_HPGe_days)
    Pb212_HPGe_LN = [0,-1.186187637,-1.637360743,-2.704583135,-3.179791882,-4.288394263,-5.686809985,-6.324607071,-6.4803616,-6.572082723,-6.612289143]
    Pb212_HPGe_LN = np.array(Pb212_HPGe_LN)
    Pb212_HPGe_LN_err = [0,0.03380838,0.058477169,0.095088079,0.141779912,0.235356211,0.362577762,0.39223548,0.307198602,0.326166989,0.334826276]
    # Literature Pb212 half-life
    Pb212_HPGe_t1half_lit = 10.628  # hours
    Pb212_HPGe_halflives = (Pb212_HPGe_days * 24) / Pb212_HPGe_t1half_lit
    Pb212_HPGe_halflives = np.array(Pb212_HPGe_halflives)


    #Three Pb curves together
    #Optimized - no blood
    #Cleared blood
    #Gamma Counter
    #Give in terms of half-ves
    xspace = np.linspace(0,  Pb212_CBlood_halflives[-1], 100)
    scattersize = 1
    fitpltPb_triple, fitaxPb_triple = plt.subplots(layout='constrained',figsize=(8,12))
    fitaxPb_triple.scatter(Pb212_GC_halflives, Pb212_GC_LN, linewidth=2.5, label='Gamma Counter', color='red',s=100, marker='^')
    #fitaxPb_triple.plot(xspace, LnCurve(xspace, *fit_paramPb_GC_hl), color='red',
                         #linestyle='dashed',linewidth=1)
    fitaxPb_triple.errorbar(Pb212_GC_halflives, Pb212_GC_LN, yerr=Pb212_GC_LN_err,xerr=None, color='red', ls='none',
                             capsize=3, capthick=1,
                             ecolor='red')
    fitaxPb_triple.scatter(Pb212_HPGe_halflives, Pb212_HPGe_LN, linewidth=2.5, label='HPGe', color='Black', s=100,
                           marker='D')
    # fitaxPb_triple.plot(xspace, LnCurve(xspace, *fit_paramPb_GC_hl), color='red',
    # linestyle='dashed',linewidth=1)
    fitaxPb_triple.errorbar(Pb212_HPGe_halflives, Pb212_HPGe_LN, yerr=Pb212_HPGe_LN_err, xerr=None, color='Black', ls='none',
                            capsize=3, capthick=1,
                            ecolor='Black')
    fitaxPb_triple.scatter(Pb212_halflives, Pb212_LN, linewidth=2.5, label='Optimized LSC', color='blue',s=100)
    #fitaxPb_triple.plot(xspace, LnCurve(xspace, *fit_paramPb_hl), color='blue',
                         #linestyle='dashed',linewidth=1)
    fitaxPb_triple.errorbar(Pb212_halflives, Pb212_LN, yerr=Pb212_LN_err,xerr=None, color='blue', ls='none',
                             capsize=3, capthick=1,
                             ecolor='blue')
    fitaxPb_triple.scatter(Pb212_CBlood_halflives, Pb212_CBlood_LN, linewidth=2.5, label='Cleared Blood LSC', color='green',s=100,marker='s')
    #fitaxPb_triple.plot(xspace, LnCurve(xspace, *fit_paramPb_CBlood_hl), color='green',
                         #linestyle='dashed',linewidth=1)
    fitaxPb_triple.errorbar(Pb212_CBlood_halflives, Pb212_CBlood_LN, yerr=Pb212_CBlood_LN_err,xerr=None, color='green', ls='none',
                             capsize=3, capthick=1,
                             ecolor='green')
    fitaxPb_triple.set_xlabel('Half-Lives Since First Sample',fontsize=26)
    fitaxPb_triple.set_ylabel('Ln(CPS/CPS$_{0}$)',fontsize=26)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    fitaxPb_triple.legend(loc='upper right',fontsize=22)
    #plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_GC_DecayCurve_HalfLife.eps",
                #format='eps')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_Combined_DecayCurve_HalfLife.png",
                format='png')
    fitpltPb_triple.show()

    #2 Pb curves together
    #Cleared blood
    #HPGe
    #Give in terms of half-ves
    xspace = np.linspace(0,  Pb212_CBlood_halflives[-1], 100)
    scattersize = 1
    fitpltPb_triple, fitaxPb_triple = plt.subplots(layout='constrained',figsize=(6,6))
    fitaxPb_triple.scatter(Pb212_HPGe_halflives, Pb212_HPGe_LN, linewidth=2.5, label='Gamma Counter', color='red',s=15, marker='^')
    #fitaxPb_triple.plot(xspace, LnCurve(xspace, *fit_paramPb_HPGe_hl), color='red',
                         #linestyle='dashed',linewidth=1)
    fitaxPb_triple.errorbar(Pb212_HPGe_halflives, Pb212_HPGe_LN, yerr=Pb212_HPGe_LN_err,xerr=None, color='red', ls='none',
                             capsize=3, capthick=1,
                             ecolor='red')
    fitaxPb_triple.scatter(Pb212_CBlood_halflives, Pb212_CBlood_LN, linewidth=2.5, label='Cleared Blood LSC', color='green',s=15,marker='s')
    #fitaxPb_triple.plot(xspace, LnCurve(xspace, *fit_paramPb_CBlood_hl), color='green',
                         #linestyle='dashed',linewidth=1)
    fitaxPb_triple.errorbar(Pb212_CBlood_halflives, Pb212_CBlood_LN, yerr=Pb212_CBlood_LN_err,xerr=None, color='green', ls='none',
                             capsize=3, capthick=1,
                             ecolor='green')
    fitaxPb_triple.set_xlabel('Half-Lives Since First Sample',fontsize=20)
    fitaxPb_triple.set_ylabel('Ln(CPS/CPS$_{0}$)',fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.gca().set_box_aspect(1)

    fitaxPb_triple.legend(loc='upper right',fontsize=18)
    #plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_HPGe_DecayCurve_HalfLife.eps",
                #format='eps')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_Combined_DecayCurve_HalfLife.png",
                format='png')
    fitpltPb_triple.show()



    #*********************************************
    # Mouse urine
    # Free Pb-212 24 hrs post injection. Mixed linear fit
    FreePb212_24hrs_hours = [0,1.05,2.066666667,3.216666666,4.616666667,5.683333333]
    FreePb212_24hrs_LNCPS = [0,-0.558656805,-1.021973177,-1.431056795,-1.764695143,-1.938435922]
    FreePb212_24hrs_CPS = [11.21888889,6.416944444,4.0375,2.681944444,1.921111111,1.614722222]
    FreePb212_24hrs_CPSErr = [0.05582435,0.042219481,0.033489219,0.027294405,0.023100692,0.02117862]

    paramFPb = [10,10]

    fit_paramFPb, fit_covFPb = curve_fit(Bi212TransEq, xdata = FreePb212_24hrs_hours, ydata = FreePb212_24hrs_CPS, maxfev = 100000)

    print('')
    print('Free Pb212 24 hrs post')
    print('Activity 1: ', fit_paramFPb[0])
    print('Half-life 1: ', np.log(2)/fit_paramFPb[1], ' hours')
    print('A2/A1: ', fit_paramFPb[1]/fit_paramFPb[0])
    print('Atomic ratio: ', (fit_paramFPb[1]/(np.log(2)/(60.551/60)))/(fit_paramFPb[0]/(np.log(2)/(10.628))))
    #print('Activity 2: ', fit_paramFPb[2])
   # print('Half-life 2: ', np.log(2)/fit_paramFPb[3], ' hours')

    xspace = np.linspace(0,  FreePb212_24hrs_hours[-1], 100)

    fitpltFPb, fitaxFPb = plt.subplots(layout='constrained')
    fitaxFPb.scatter(FreePb212_24hrs_hours, FreePb212_24hrs_CPS, linewidth=2.5, label='Free $^{212}$Pb', color='black')
    fitaxFPb.errorbar(FreePb212_24hrs_hours, FreePb212_24hrs_CPS, yerr=FreePb212_24hrs_CPSErr, color='black', ls='none', capsize=3, capthick=1,
                        ecolor='black')
    fitaxFPb.plot(xspace, Bi212TransEq(xspace, *fit_paramFPb), label='Composite Activity', color='red', linestyle='dashed')
    fitaxFPb.plot(xspace,PbActivity(xspace, fit_paramFPb[0]), label='Pb Decay', color='blue', linestyle='dashed')
    fitaxFPb.plot(xspace,BiActivity(xspace, fit_paramFPb[1]), label='Bi Decay', color='green', linestyle='dashed')
    fitaxFPb.plot(xspace,BiGrowth(xspace, fit_paramFPb[0]), label='Bi Growth', color='purple', linestyle='dashed')


    fitaxFPb.set_yscale('log')
    fitaxFPb.set_xlabel('Time Since First Sample (Hours)')
    fitaxFPb.set_ylabel('Activity (CPS)')
    fitaxFPb.legend(loc='lower right')


    fitpltFPb.show()

    # Free Pb-212 48 hrs post injection.
    FreePb212_48hrs_hours = [0,1,2,3.016666667,4.016666667]
    FreePb212_48hrs_LNCPS = [0,-0.541953148,-1.013205473,-1.374945628,-1.659655767]
    FreePb212_48hrs_CPS = [20.05444444,11.66388889,7.280833333,5.070833333,3.814444444]
    FreePb212_48hrs_CPSErr = [0.074636982,0.056920727,0.044971699,0.037530852,0.032551005]

    paramFPb = [10, 10]

    fit_paramFPb48, fit_covFPb48 = curve_fit(Bi212TransEq, xdata=FreePb212_48hrs_hours, ydata=FreePb212_48hrs_CPS,
                                         maxfev=100000)

    print('')
    print('Free Pb212 24 hrs post')
    print('Activity 1: ', fit_paramFPb48[0])
    print('Half-life 1: ', np.log(2) / fit_paramFPb48[1], ' hours')
    print('A2/A1: ', fit_paramFPb48[1] / fit_paramFPb48[0])
    print('Atomic ratio: ',
          (fit_paramFPb48[1] / (np.log(2) / (60.551 / 60))) / (fit_paramFPb48[0] / (np.log(2) / (10.628))))
    # print('Activity 2: ', fit_paramFPb[2])
    # print('Half-life 2: ', np.log(2)/fit_paramFPb[3], ' hours')

    xspace = np.linspace(0, FreePb212_48hrs_hours[-1], 100)

    fitpltFPb48, fitaxFPb48 = plt.subplots(layout='constrained')
    fitaxFPb48.scatter(FreePb212_48hrs_hours, FreePb212_48hrs_CPS, linewidth=2.5, label='Free $^{212}$Pb', color='black')
    fitaxFPb48.errorbar(FreePb212_48hrs_hours, FreePb212_48hrs_CPS, yerr=FreePb212_48hrs_CPSErr, color='black', ls='none',
                      capsize=3, capthick=1,
                      ecolor='black')
    fitaxFPb48.plot(xspace, Bi212TransEq(xspace, *fit_paramFPb48), label='Composite Activity', color='red',
                  linestyle='dashed')
    fitaxFPb48.plot(xspace, PbActivity(xspace, fit_paramFPb48[0]), label='Pb Decay', color='blue', linestyle='dashed')
    fitaxFPb48.plot(xspace, BiActivity(xspace, fit_paramFPb48[1]), label='Bi Decay', color='green', linestyle='dashed')
    fitaxFPb48.plot(xspace, BiGrowth(xspace, fit_paramFPb48[0]), label='Bi Growth', color='purple', linestyle='dashed')

    fitaxFPb48.set_yscale('log')
    fitaxFPb48.set_xlabel('Time Since First Sample (Hours)')
    fitaxFPb48.set_ylabel('Activity (CPS)')
    fitaxFPb48.legend(loc='lower right')

    fitpltFPb48.show()


main()