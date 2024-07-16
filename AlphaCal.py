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

def QuenchCal(x, a, b, c, d, e):

    return a * (x -d) + b * np.log(1 + c * (x - d)) + e

def main():

    # Make separate arrays
    alphaE_Ra = [5.6802, 6.6744, 7.3861]
    alphaCH_Ra = [1745.8529, 2157.4374, 2561.8918]

    alphaE_Ac = [5.8089, 6.3081, 7.0669, 8.376]
    alphaCH_Ac = [1768.6443, 2018.5875, 2462.4987, 3295.8456]

    alphaE = [5.6802, 6.6744, 7.3861, 5.8089, 6.3081, 7.0669, 8.376]
    alphaCH = [1745.8529, 2157.4374, 2561.8918, 1768.6443, 2018.5875, 2462.4987, 3295.8456]

    aplt, ax = plt.subplots(layout = 'constrained')
    ax.scatter(alphaE_Ra, alphaCH_Ra, linewidth = 2.5, label='Ra-223 Decay Chain')
    ax.scatter(alphaE_Ac, alphaCH_Ac, linewidth = 2.5, label='Ac-225 Decay Chain')
    ax.set_xlabel('Decay Energy (MeV)')
    ax.set_ylabel('ADC Channel')

    ax.legend(loc='upper left')
    aplt.show()

    param = [500, -100, 0.000005, 1, 1]

    fit_param, fit_cov = curve_fit(QuenchCal, xdata = alphaE, ydata = alphaCH, p0 = param, bounds=((0, -np.inf, -np.inf, -np.inf, -np.inf),(np.inf, np.inf, np.inf, np.inf, np.inf)), maxfev = 100000)

    print(fit_param)

    xspace = np.linspace(5.6802,8.376, 100)

    fitplt, fitax = plt.subplots(layout = 'constrained')
    fitax.scatter(alphaE_Ra, alphaCH_Ra, linewidth = 2.5, label='Ra-223 Decay Chain')
    fitax.scatter(alphaE_Ac, alphaCH_Ac, linewidth = 2.5, label='Ac-225 Decay Chain')
    fitax.plot(xspace,QuenchCal(xspace,*fit_param), color='green')
    fitax.set_xlabel('Decay Energy (MeV)')
    fitax.set_ylabel('ADC Channel')

    fitax.legend(loc='upper left')
    fitplt.show()

    # Calibrate the Pb212 peaks with optical grease
    alphaE_Pb = [5.607, 5.768, 6.0624, 8.78486]
    alphaCH_Pb = [758.8254, 1175.9632, 1869.8444, 3389.3069]

    paramPb = [500, 100, 0.00000005, 1, 1]

    fit_paramPb, fit_covPb = curve_fit(QuenchCal, xdata = alphaE_Pb, ydata = alphaCH_Pb, p0 = paramPb, bounds=((0, -np.inf, -np.inf, -np.inf, -np.inf),(np.inf, np.inf, np.inf, np.inf, np.inf)), maxfev = 100000)

    print(fit_paramPb)

    xspace = np.linspace(5.607,8.78486, 100)

    fitpltPb, fitaxPb = plt.subplots(layout = 'constrained')
    fitaxPb.scatter(alphaE_Pb, alphaCH_Pb, linewidth = 2.5, label='Pb-212 Decay Chain')
    fitaxPb.plot(xspace,QuenchCal(xspace,*fit_paramPb), color='green')
    fitaxPb.set_xlabel('Decay Energy (MeV)')
    fitaxPb.set_ylabel('ADC Channel')

    fitaxPb.legend(loc='upper left')
    fitpltPb.show()

    # Plot Pb212 and Ra223+Ac225
    fitplt_Pb_RaAc, fitax_Pb_RaAc = plt.subplots(layout = 'constrained')
    fitax_Pb_RaAc.scatter(alphaE_Pb, alphaCH_Pb, linewidth = 2.5, color='green', label='Pb-212 Decay Chain')
    fitax_Pb_RaAc.plot(xspace,QuenchCal(xspace,*fit_paramPb), color='green', linestyle='dashed')
    fitax_Pb_RaAc.scatter(alphaE, alphaCH, linewidth = 2.5, color='Blue', label='Ra-223 and Ac-225 Decay Chains')
    fitax_Pb_RaAc.plot(xspace,QuenchCal(xspace,*fit_param), color='Blue', linestyle='dashed')
    fitax_Pb_RaAc.set_xlabel('Decay Energy (MeV)')
    fitax_Pb_RaAc.set_ylabel('ADC Channel')

    fitax_Pb_RaAc.legend(loc='upper left')
    fitplt_Pb_RaAc.show()

    # Plot Pb212 and Ra223+Ac225 - flip axes
    fitplt_Pb_RaAc, fitax_Pb_RaAc = plt.subplots(layout = 'constrained')
    fitax_Pb_RaAc.scatter(alphaCH_Pb, alphaE_Pb, linewidth = 2.5, color='green', label='Pb-212 Decay Chain')
    fitax_Pb_RaAc.plot(QuenchCal(xspace,*fit_paramPb), xspace, color='green', linestyle='dashed')
    fitax_Pb_RaAc.scatter(alphaCH, alphaE, linewidth = 2.5, color='Blue', label='Ra-223 and Ac-225 Decay Chains')
    fitax_Pb_RaAc.plot(QuenchCal(xspace,*fit_param), xspace, color='Blue', linestyle='dashed')
    fitax_Pb_RaAc.set_ylabel('Decay Energy (MeV)')
    fitax_Pb_RaAc.set_xlabel('ADC Channel')

    fitax_Pb_RaAc.legend(loc='upper left')
    fitplt_Pb_RaAc.show()

    # Calibration for channels to E
    paramPb_flipped = [500, -100, -0.00000005, 1, 1]

    fit_paramPb_flipped, fit_covPb_flipped = curve_fit(QuenchCal, xdata = alphaCH_Pb, ydata = alphaE_Pb, p0 = paramPb_flipped, bounds=((0, -np.inf, -np.inf, -np.inf, -np.inf),(np.inf, np.inf, np.inf, np.inf, np.inf)), maxfev = 100000)

    # Calculate the energy of the first peak
    Pb_peak1E = QuenchCal(412.662, *fit_paramPb_flipped)
    print(Pb_peak1E)

main()