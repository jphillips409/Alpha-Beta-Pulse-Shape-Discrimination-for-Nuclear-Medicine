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

plt.rcParams['font.size'] = 15
plt.rcParams['axes.titlepad'] = 15

def LinCurve(x, a, t):

    return a * np.exp(-t*x)

def LnCurve(x, t):

    return -t*x

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

    # Same for patient 2
    XofigoRaw_CPS_p2 = np.array([14.62944444,26.75902778,20.43041667])
    XofigoRaw_CPS_Un_p2 = np.array([0.090152477,0.060963363,0.053268733])
    XofigoRaw_CPS_RUn_p2 = XofigoRaw_CPS_Un_p2/XofigoRaw_CPS_p2
    # Measurement time post injection, days
    XofigoRaw_TimePost_p2 = np.array([0.025694444,5.765277778,5.85625])
    # Saliva weight (g)
    XofigoSalWeight_p2 = np.array([0.2041,0.2565,0.2433])

    # Get initial activity
    XofigoCorr_CPS_p2 = RaInitialActivity(XofigoRaw_TimePost_p2,XofigoRaw_CPS_p2)
    # Get specific activity
    Xofigo_SpecCPS_p2 = XofigoCorr_CPS_p2/XofigoSalWeight_p2
    Xofigo_SpecCPS_Un_p2 = XofigoRaw_CPS_RUn_p2*Xofigo_SpecCPS_p2
    print(Xofigo_SpecCPS_Un_p2)
    # Time since extraction (days)
    Xofigo_ExtractTime_p2 = np.array([0.003472222,0.006944444,0.010416667])

    fit_InjecCurve, ax_InjecCurve = plt.subplots()
    ax_InjecCurve.scatter(Xofigo_ExtractTime*24*60, Xofigo_SpecCPS, linewidth = 2.5, color='blue',label='Patient 1')
    ax_InjecCurve.errorbar(Xofigo_ExtractTime*24*60, Xofigo_SpecCPS, yerr=Xofigo_SpecCPS_Un, color='blue', ls='none',  capsize=3, capthick=1, ecolor='blue')
    ax_InjecCurve.scatter(Xofigo_ExtractTime_p2 * 24 * 60, Xofigo_SpecCPS_p2 , linewidth=2.5, color='red',label='Patient 2')
    ax_InjecCurve.errorbar(Xofigo_ExtractTime_p2  * 24 * 60, Xofigo_SpecCPS_p2 , yerr=Xofigo_SpecCPS_Un_p2 , color='red',ls='none', capsize=3, capthick=1, ecolor='red')
    ax_InjecCurve.set_xlabel('Time Since Injection (min)',fontsize=20)
    ax_InjecCurve.set_ylabel('Specific Activity (CPS/g)',fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0,160)
    plt.xlim(0,20)
    plt.xticks(np.arange(0,20.01,5))
    ax_InjecCurve.legend(loc='lower right')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ra223_XofigoInjec.eps", format='eps')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ra223_XofigoInjec.png", format='png')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ra223_XofigoInjec.svg", format='svg')
    plt.show()

    # Plot Sample D (use as initial) and Sample A (use as final)
    # MUST USE SPECIFIC ACTIVITIES
    # Assume exactly four weeks between A and D

    Final_CPS = 0.031230159
    Final_CPS_err = 0.001113235
    Final_CPS_relerr = Final_CPS_err/Final_CPS
    Final_CorrCPS = RaInitialActivity(1.828472222,Final_CPS)
    Final_SpecCPS = Final_CorrCPS/0.2435

    Xofigo_HLtimes = np.array([0,28])
    Xofigo_HLCPS = np.array([Xofigo_SpecCPS[2],Final_SpecCPS])
    Xofigo_HLCPS_log = np.array([0,np.log(Xofigo_HLCPS[1]/Xofigo_HLCPS[0])])
    fit_paramEffHL, fit_covEffHL = curve_fit(RaDecayWithBio, xdata = Xofigo_HLtimes, ydata = Xofigo_HLCPS_log, maxfev = 100000)
    print('****************************************')
    print('Patient 1')
    print('Initial activity: ', Xofigo_HLCPS[0], ' Activity after 4 weeks: ', Xofigo_HLCPS[1])
    print('Extracted Bio HL: ', fit_paramEffHL[0])
    print('Effective HL: ', (11.4352*fit_paramEffHL[0])/(11.4352+fit_paramEffHL[0]))
    xspace = np.linspace(0, Xofigo_HLtimes[-1], 100)

    fitpltBioHL, fitaxBioHL = plt.subplots()
    fitaxBioHL.scatter(Xofigo_HLtimes, Xofigo_HLCPS_log, linewidth=2.5, label='Data',
                       color='black')
    #fitaxBioHL.errorbar(Xofigo_HLtimes, Xofigo_HLCPS_log, yerr=FreePb212_48hrs_CPSErr, color='black',
                        #ls='none',
                        #capsize=3, capthick=1,
                        #ecolor='black')
    fitaxBioHL.plot(xspace, LnCurve(xspace, np.log(2)/11.4352), label=r'Physical $t_{1/2}$', color='blue',
                    linestyle='dashed')
    fitaxBioHL.plot(xspace, LnCurve(xspace, np.log(2)/3.750151), label=r'Partial Biological $t_{1/2}$', color='red',
                    linestyle='dashed')
    fitaxBioHL.plot(xspace, RaDecayWithBio(xspace, *fit_paramEffHL), label=r'Total Biological $t_{1/2}$', color='green',
                    linestyle='dashed')
    fitaxBioHL.set_xlabel('Time (days)',fontsize=20)
    fitaxBioHL.set_ylabel(r'Ln(CPS/CPS$_{0}$)',fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.ylim(-8,0)
    fitaxBioHL.legend(loc='lower left')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ra223_Patient1_BioHL.eps", format='eps')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ra223_Patient1_BioHL.png", format='png')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ra223_Patient1_BioHL.svg", format='svg')
    plt.show()

    print('LN uncertainty of initial and final: ', np.sqrt(Final_CPS_relerr**2 + (Xofigo_SpecCPS_Un[2]/Xofigo_SpecCPS[2])**2))
    print('Relative version: ', np.sqrt(Final_CPS_relerr**2 + (Xofigo_SpecCPS_Un[2]/Xofigo_SpecCPS[2])**2)/(np.log(Xofigo_HLCPS[1]/Xofigo_HLCPS[0])))
    print('Absolute uncertainty of effective half-life: ',0.005196887713449432*2.824019333204879)

    numerator = 2.824019333204879*11.4352
    num_relerr = np.sqrt((0.0010/11.4352)**2 + 0.005196887713449432**2)

    denom = 11.4352 - 2.824019333204879
    denom_relerr = np.sqrt(0.014676111375276094**2 + 0.0010**2)/denom

    BioHL_relerr = np.sqrt(num_relerr**2 + denom_relerr**2)
    BioHL_err = BioHL_relerr*3.7501507782304158

    print('Bio relative HL: ', BioHL_relerr)
    print('Absolute version: ',BioHL_err)

    # Same for patient 2
    # Use samples E and G (H is bad)
    Final_CPS_p2 = 0.010555556
    Final_CPS_err_p2 = 0.000605403
    Final_CPS_relerr_p2 = Final_CPS_err_p2 / Final_CPS_p2
    Final_CorrCPS_p2 = RaInitialActivity(5.763888889, Final_CPS_p2)
    Final_SpecCPS_p2 = Final_CorrCPS_p2 / 0.2403

    Xofigo_HLtimes_p2 = np.array([0, 28])
    Xofigo_HLCPS_p2 = np.array([Xofigo_SpecCPS_p2[1], Final_SpecCPS_p2])
    Xofigo_HLCPS_log_p2 = np.array([0, np.log(Xofigo_HLCPS_p2[1] / Xofigo_HLCPS_p2[0])])
    fit_paramEffHL_p2, fit_covEffHL_p2 = curve_fit(RaDecayWithBio, xdata=Xofigo_HLtimes_p2, ydata=Xofigo_HLCPS_log_p2,
                                             maxfev=100000)
    print('****************************************')
    print('Patient 2')
    print('Initial activity: ', Xofigo_HLCPS_p2[0], ' Activity after 4 weeks: ', Xofigo_HLCPS_p2[1])
    print('Extracted Bio HL: ', fit_paramEffHL_p2[0])
    print('Effective HL: ', (11.4352 * fit_paramEffHL_p2[0]) / (11.4352 + fit_paramEffHL_p2[0]))
    xspace = np.linspace(0, Xofigo_HLtimes_p2[-1], 100)

    fitpltBioHL_p2, fitaxBioHL_p2 = plt.subplots()
    fitaxBioHL_p2.scatter(Xofigo_HLtimes_p2, Xofigo_HLCPS_log_p2, linewidth=2.5, label='Data',
                       color='black')
    # fitaxBioHL.errorbar(Xofigo_HLtimes, Xofigo_HLCPS_log, yerr=FreePb212_48hrs_CPSErr, color='black',
    # ls='none',
    # capsize=3, capthick=1,
    # ecolor='black')
    fitaxBioHL_p2.plot(xspace, LnCurve(xspace, np.log(2) / 11.4352), label=r'Physical $t_{1/2}$', color='blue',
                    linestyle='dashed')
    fitaxBioHL_p2.plot(xspace, LnCurve(xspace, np.log(2) / 3.194), label=r'Partial Biological $t_{1/2}$', color='red',
                    linestyle='dashed')
    fitaxBioHL_p2.plot(xspace, RaDecayWithBio(xspace, *fit_paramEffHL_p2), label=r'Total Biological $t_{1/2}$', color='green',
                    linestyle='dashed')
    fitaxBioHL_p2.set_xlabel('Time (days)', fontsize=20)
    fitaxBioHL_p2.set_ylabel(r'Ln(CPS/CPS$_{0}$)', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    fitaxBioHL_p2.legend(loc='lower left')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ra223_Patient2_BioHL.eps", format='eps')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ra223_Patient2_BioHL.png", format='png')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ra223_Patient2_BioHL.svg", format='svg')
    plt.show()

    print('LN uncertainty of initial and final: ',
          np.sqrt(Final_CPS_relerr ** 2 + (Xofigo_SpecCPS_Un_p2[1] / Xofigo_SpecCPS_p2[1]) ** 2))
    print('Relative version: ', np.sqrt(Final_CPS_relerr_p2 ** 2 + (Xofigo_SpecCPS_Un_p2[1] / Xofigo_SpecCPS_p2[1]) ** 2) / (
        np.log(Xofigo_HLCPS_p2[1] / Xofigo_HLCPS_p2[0])))
    print('Absolute uncertainty of effective half-life: ', 0.007384605298755379 * 2.496921943499092)

    numerator = 2.496921943499092 * 11.4352
    num_relerr = np.sqrt((0.0010 / 11.4352) ** 2 + 0.007384605298755379 ** 2)

    denom = 11.4352 - 2.496921943499092
    denom_relerr = np.sqrt(0.018438783014541976 ** 2 + 0.0010 ** 2) / denom

    BioHL_relerr = np.sqrt(num_relerr ** 2 + denom_relerr ** 2)
    BioHL_err = BioHL_relerr * 3.19444099051428

    print('Bio relative HL: ', BioHL_relerr)
    print('Absolute version: ', BioHL_err)

    # Plot the two patient curves together
    fitpltBioHL_all, fitaxBioHL_all = plt.subplots()
    fitaxBioHL_all.scatter(Xofigo_HLtimes, Xofigo_HLCPS_log, linewidth=2.5,color='blue')
    fitaxBioHL_all.plot(xspace, RaDecayWithBio(xspace, *fit_paramEffHL), label=r'Patient 1', color='blue',linestyle='dashed')
    fitaxBioHL_all.scatter(Xofigo_HLtimes_p2, Xofigo_HLCPS_log_p2, linewidth=2.5,color='red')
    fitaxBioHL_all.plot(xspace, RaDecayWithBio(xspace, *fit_paramEffHL_p2), label=r'Patient 2', color='red',linestyle='dashed')
    fitaxBioHL_all.set_xlabel('Time (days)', fontsize=20)
    fitaxBioHL_all.set_ylabel(r'Ln(CPS/CPS$_{0}$)', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    fitaxBioHL_all.legend(loc='lower left')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ra223_AllPatients_BioHL.eps", format='eps')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ra223_AllPatients_BioHL.png", format='png')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ra223_AllPatients_BioHL.svg", format='svg')
    plt.show()

    # Tracking the decay of the pre-injection samples
    XofigoP1_Pre_Days = np.array([0,6.001388889,16.99027778,27.99722222])
    XofigoP1_Pre_LN = np.array([0,-0.27829983,-0.887453627,-1.444515136])
    XofigoP1_Pre_LN_err = np.array([0,0.010321411,0.04453873,0.096515727])
    # Fit to LN curve
    xspace = np.linspace(0, XofigoP1_Pre_Days[-1], 100)
    fit_paramP1_Pre, fit_covP1_Pre = curve_fit(LnCurve, xdata = XofigoP1_Pre_Days, ydata = XofigoP1_Pre_LN, maxfev = 100000)

    fitpltP1_Pre, fitaxP1_Pre = plt.subplots(figsize=(6.4,10.0997))
    fitaxP1_Pre.scatter(XofigoP1_Pre_Days, XofigoP1_Pre_LN, linewidth=2.5, label='Data',
                       color='blue')
    fitaxP1_Pre.errorbar(XofigoP1_Pre_Days, XofigoP1_Pre_LN, yerr=XofigoP1_Pre_LN_err, color='blue',
                        ls='none',
                        capsize=3, capthick=1,
                        ecolor='blue')
    fitaxP1_Pre.plot(xspace, LnCurve(xspace, fit_paramP1_Pre), color='blue',
                    linestyle='dashed')
    fitaxP1_Pre.set_xlabel('Time (days)',fontsize=20)
    fitaxP1_Pre.set_ylabel(r'Ln(CPS/CPS$_{0}$)',fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.ylim(-8,0)
    #fitaxP1_Pre.legend(loc='lower left')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ra223_DecayCurve_PreInject.eps", format='eps')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ra223_DecayCurve_PreInject.png", format='png')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Ra223_DecayCurve_PreInject.svg", format='svg')
    plt.show()
main()