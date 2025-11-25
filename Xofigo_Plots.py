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

# Find inital activity
# HL in days
def RaInitialActivity(x,A):
    return A/np.exp(-(np.log(2)/11.4352)*x)

def RaDecay(x,A):
    return A*np.exp(-(np.log(2)/11.4352)*x)

# Ln curve
def RaDecayWithBio(x,t):
    # Uses effective HL
    return -(np.log(2)*(11.4352 + t)/(11.4352*t))*x

# For transient equilibrium, t1 and t2 are not parameters
def Bi212TransEq(x,A1,A2):
    t1 = np.log(2)/10.628 # Pb-212 decay constant
    t2 = np.log(2)/(60.551/60) # Bi-212 decay constant
    return A1*np.exp(-t1*x) + (t2/(t2-t1))*A1*(np.exp(-t1*x)-np.exp(-t2*x)) + A2*np.exp(-t2*x)

def main():

    # Plot the growth of 223Ra activity in the saliva since injection
    # Decay correct the counts
    # 3 samples: B, C, D
    XofigoRaw_CPS = np.array([13.93069444,22.69319444,27.98472222])
    XofigoRaw_CPS_Un = np.array([0.043986574,0.056141184,0.062343941])
    XofigoRaw_CPS_RUn = XofigoRaw_CPS_Un/XofigoRaw_CPS
    # Measurement time post injection, days
    XofigoRaw_TimePost = np.array([0.902777778,0.807638889,0.05625])
    # Saliva weight (g)
    XofigoSalWeight = np.array([0.1546,0.169,0.203])

    # Get initial activity
    XofigoCorr_CPS = RaInitialActivity(XofigoRaw_TimePost,XofigoRaw_CPS)
    # Get specific activity
    Xofigo_SpecCPS = XofigoCorr_CPS/XofigoSalWeight
    Xofigo_SpecCPS_Un = XofigoRaw_CPS_RUn*Xofigo_SpecCPS
    print(Xofigo_SpecCPS_Un)
    # Time since extraction (days)
    Xofigo_ExtractTime = np.array([0.003472222,0.006944444,0.010416667])

    fit_InjecCurve, fit_InjecCurve = plt.subplots(layout = 'constrained')
    fit_InjecCurve.scatter(Xofigo_ExtractTime*24*60, Xofigo_SpecCPS, linewidth = 2.5, color='blue')
    fit_InjecCurve.errorbar(Xofigo_ExtractTime*24*60, Xofigo_SpecCPS, yerr=Xofigo_SpecCPS_Un, color='blue', ls='none',  capsize=3, capthick=1, ecolor='black')
    fit_InjecCurve.set_xlabel('Time Since Injection (min)',fontsize=20)
    fit_InjecCurve.set_ylabel('Specific Activity (CPS/g)',fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0,160)
    plt.xlim(0,20)
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ra223_XofigoInjec.eps", format='eps')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ra223_XofigoInjec.png", format='png')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ra223_XofigoInjec.svg", format='svg')
    plt.show()

    # Plot Sample D (use as initial) and Sample A (use as final)
    # Assume exactly four weeks between A and D
    Xofigo_HLtimes = np.array([0,28])
    Xofigo_HLCPS = np.array([XofigoCorr_CPS[2],0.128255272])
    Xofigo_HLCPS_log = np.array([0,np.log(Xofigo_HLCPS[1]/Xofigo_HLCPS[0])])
    fit_paramEffHL, fit_covEffHL = curve_fit(RaDecayWithBio, xdata = Xofigo_HLtimes, ydata = Xofigo_HLCPS_log, maxfev = 100000)
    print('Initial activity: ', Xofigo_HLCPS[0], ' Activity after 4 weeks: ', Xofigo_HLCPS[1])
    print('Extracted Bio HL: ', fit_paramEffHL[0])
    print('Effective HL: ', (11.4352*fit_paramEffHL[0])/(11.4352+fit_paramEffHL[0]))
    xspace = np.linspace(0, Xofigo_HLtimes[-1], 100)

    fitpltBioHL, fitaxBioHL = plt.subplots(layout='constrained')
    fitaxBioHL.scatter(Xofigo_HLtimes, Xofigo_HLCPS_log, linewidth=2.5, label='Data',
                       color='black')
    #fitaxBioHL.errorbar(Xofigo_HLtimes, Xofigo_HLCPS_log, yerr=FreePb212_48hrs_CPSErr, color='black',
                        #ls='none',
                        #capsize=3, capthick=1,
                        #ecolor='black')
    fitaxBioHL.plot(xspace, RaDecayWithBio(xspace, *fit_paramEffHL), label='Effective Decay Curve', color='red',
                    linestyle='dashed')
    fitaxBioHL.set_xlabel('Time (days)')
    fitaxBioHL.set_ylabel('Ln(CPS_{0}/CPS)')
    fitaxBioHL.legend(loc='lower right')

    plt.show()
main()