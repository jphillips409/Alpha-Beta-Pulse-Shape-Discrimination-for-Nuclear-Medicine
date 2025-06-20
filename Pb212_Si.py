########################################################
# Created on July 10 2024
# @author: Johnathan Phillips
# @email: j.s.phillips@wustl.edu

# Purpose:

# Detector:
########################################################

import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.optimize import curve_fit
from scipy.special import erfc
from scipy.integrate import quad
import matplotlib as mpl
from pathlib import Path

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

#the response to an alpha particle can be modeled as a exponential tail convoluted with a gaussian
#(https://doi.org/10.1016/j.apradiso.2014.11.023) Improved peak shape fitting in alpha spectra, S. Pommé
def peakshape(ch, A, mu, sigma, tau):
    tail = A/(2*tau)*np.exp((ch-mu)/tau + sigma**2/(2*tau**2))
    gaus = erfc(((ch-mu)/sigma+sigma/tau)/np.sqrt(2))
    return gaus*tail

def peakshape_triptail(ch, A, mu, sigma, tau1, tau2, tau3):
    tail1 = A/(2*tau1)*np.exp((ch-mu)/tau1 + sigma**2/(2*tau1**2))
    tail2 = A/(2*tau2)*np.exp((ch-mu)/tau2 + sigma**2/(2*tau2**2))
    tail2 = A/(2*tau3)*np.exp((ch-mu)/tau3 + sigma**2/(2*tau3**2))
    gaus = erfc(((ch-mu)/sigma+sigma/tau)/np.sqrt(2))
    return gaus*tail1*tail2*tail3

def double_peakshape(ch, A1, mu1, sigma1, tau1, A2, mu2, sigma2, tau2):
    return peakshape(ch, A1, mu1, sigma1, tau1) + peakshape(ch, A2, mu2, sigma2, tau2)

def triple_peakshape(ch, A, mu, sigma, tau1, tau2, tau3):
    return peakshape(ch, A, mu, sigma, tau1) + peakshape(ch, A, mu, sigma, tau2) + peakshape(ch, A, mu, sigma, tau3)

def two_triple_peakshape(ch, A1, mu1, sigma1, tau1_1, tau1_2, tau1_3, A2, mu2, sigma2, tau2_1, tau2_2, tau2_3):
    return triple_peakshape(ch, A1, mu1, sigma1, tau1_1, tau1_2, tau1_3) + triple_peakshape(ch, A2, mu2, sigma2, tau2_1, tau2_2, tau2_3)

def main():

    # Constant filepath variable to get around the problem of backslashes in windows
    # The Path library will use forward slashes but convert them to correctly treat your OS
    # Also makes it easier to switch to a different computer
    filepath = Path(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Pb212_Si")

    # This code block reads data saved from the alpha particle detector
    data_file = r'Pb212_Si_07_09_t1800_bottom.Spe'

    # Read in a second data file
    #data_file = r'Pb212_Si_07_09_t600_bottom_65V.Spe'

    file_to_open = filepath / data_file

    #file_to_open2 = filepath / data_file2


    counts = np.genfromtxt(file_to_open, skip_header=12, skip_footer=14)
    ch = np.array(range(1, counts.size + 1))

    #counts2 = np.genfromtxt(file_to_open2, skip_header=12, skip_footer=14)
   # ch2 = np.array(range(1, counts2.size + 1))

    #Calibrate the alpha data using the MAESTRO linear calibration
    energy = (ch * 3.088979) + 3493.313721
   # energy2 = (ch2 * 3.088979) + 3493.313721


    # Log Plot
    logplt, logax = plt.subplots(layout = 'constrained')
    logax.plot(ch,counts)
    logax.set_yscale('log')
    #logax.set_xlabel('Energy (keV)')
    logax.set_xlabel('Channels')
    logax.set_ylabel('Counts')
    logplt.show()

    # Lin Plot
    linplt, linax = plt.subplots(layout = 'constrained')
    linax.plot(ch,counts)
    #linax.set_xlabel('Energy (keV)')
    linax.set_xlabel('Channels')
    linax.set_ylabel('Counts')
    linplt.show()

    # Log Plot - 65 V
  #  logplt, logax = plt.subplots(layout = 'constrained')
  #  logax.plot(energy2,counts2)
   # logax.set_yscale('log')
  #  logax.set_xlabel('Energy (keV)')
   # logax.set_ylabel('Counts')
  #  logax.set_label('65 V')
  #  logplt.show()

    # Lin Plot - 65 V
  #  linplt, linax = plt.subplots(layout = 'constrained')
  #  linax.plot(energy2,counts2)
  #  linax.set_xlabel('Energy (keV)')
  # linax.set_ylabel('Counts')
  #  linax.set_label('65 V')
   # linplt.show()

    # Po212 peak in ch numbers
    Poplt, Poax = plt.subplots(layout = 'constrained')
    Poax.plot(ch,counts)
    Poax.set_xlabel('Channels')
    Poax.set_ylabel('Counts')
    plt.xlim(600,900) # if using optical grease
    Poplt.show()

    # Bi212 Plot
    Biplt, Biax = plt.subplots(layout = 'constrained')
    Biax.plot(ch,counts)
    Biax.plot(ch,peakshape(ch,80000,1630,2, 30))
    Biax.set_xlabel('Channels')
    Biax.set_ylabel('Counts')
    plt.xlim(1250,1750) # if using optical grease
    Biplt.show()

    # Fit the Po212 peaks using a convolution of two Gaussians with 3 exponentials each
    Bi212_lowerbound = 650
    Bi212_upperbound = 900
    Bi212peak = copy.deepcopy(counts[Bi212_lowerbound:Bi212_upperbound])
    Bi212E = copy.deepcopy(energy[Bi212_lowerbound:Bi212_upperbound])
    Bi212ch = np.linspace(Bi212_lowerbound,Bi212_upperbound, len(Bi212peak))
    print(len(Bi212ch))
    print(len(Bi212peak))

    # For 50 V
    Bi212popt, Bi212pcov = curve_fit(two_triple_peakshape, Bi212ch, Bi212peak, p0 = [60000, 800, 2, 80, 80, 80, 10000, 830,2, 80, 80, 80], bounds=((0,700,-np.inf,-np.inf,-np.inf,-np.inf,0,700,-np.inf,-np.inf,-np.inf,-np.inf),(np.inf, 900, np.inf, np.inf, np.inf, np.inf, np.inf, 900, np.inf, np.inf, np.inf, np.inf)) ,maxfev = 100000)
    # For 60 V
    #Bi212popt, Bi212pcov = curve_fit(two_triple_peakshape, Bi212ch, Bi212peak, p0 = [60000, 805, 2, 80, 80, 80, 10000, 840,2, 80, 80, 80], bounds=((0,790,-np.inf,-np.inf,-np.inf,-np.inf,0,790,-np.inf,-np.inf,-np.inf,-np.inf),(np.inf, 820, np.inf, np.inf, np.inf, np.inf, np.inf, 850, np.inf, np.inf, np.inf, np.inf)) ,maxfev = 100000)

    print(Bi212popt)

    # Fitted Bi212 Plot
    Biplt_fit, Biax_fit = plt.subplots(layout = 'constrained')
    Biax_fit.plot(Bi212ch,Bi212peak)
    #Biax_fit.plot(Bi212ch,two_triple_peakshape(Bi212ch,*Bi212popt), linewidth=2.5)
    Biax_fit.set_xlabel('Channels')
    Biax_fit.set_ylabel('Counts')
    Biax_fit.set_yscale('log')
    plt.ylim(1,5000) # if using optical grease
    Biplt_fit.show()

    # Fit the Po212 peak using convolution of a Gaussian with 3 exponentials
    Po212_lowerbound = 1400
    po212_upperbound = 1750
    Po212peak = copy.deepcopy(counts[Po212_lowerbound:po212_upperbound])
    Po212E = copy.deepcopy(energy[Po212_lowerbound:po212_upperbound])
    Po212ch = np.linspace(Po212_lowerbound, po212_upperbound, len(Po212peak))

    # For 50 V
    Po212popt, Po212pcov = curve_fit(triple_peakshape, Po212ch, Po212peak, p0 = [120000,1600 ,2, 80, 80, 80], bounds=((0,1550,-np.inf,-np.inf,-np.inf,-np.inf),(np.inf, 1650, np.inf, np.inf, np.inf, np.inf)), maxfev = 100000)
    # For 60 V
    #Po212popt, Po212pcov = curve_fit(triple_peakshape, Po212ch, Po212peak, p0 = [120000,1600 ,2, 80, 80, 70], bounds=((0,1550,-np.inf,-np.inf,-np.inf,-np.inf),(np.inf, 1650, np.inf, np.inf, np.inf, np.inf)), maxfev = 100000)

    print(Po212popt)

    # Fitted Po212 Plot
    Poplt_fit, Poax_fit = plt.subplots(layout = 'constrained')
    Poax_fit.plot(Po212ch,Po212peak)
    #Poax_fit.plot(Po212ch,triple_peakshape(Po212ch,*Po212popt), linewidth=2.5)
    Poax_fit.set_xlabel('Channels')
    Poax_fit.set_ylabel('Counts')
    Poax_fit.set_yscale('log')
    plt.ylim(1,5000)
    Poplt_fit.show()

    # Plots a spectrum with both
    plt_fit, ax_fit = plt.subplots(layout = 'constrained')
    ax_fit.plot(ch,counts, label=r'Alpha Spectrum')
    #ax_fit.plot(Bi212ch,two_triple_peakshape(Bi212ch,*Bi212popt), linewidth=2.5, label=r'Bi-212')
    #ax_fit.plot(Po212ch,triple_peakshape(Po212ch,*Po212popt), linewidth=2.5, label=r'Po-212')
    ax_fit.set_xlabel('Channels')
    ax_fit.set_ylabel('Counts')
    ax_fit.set_yscale('log')
    plt.ylim(1,5000)
    plt.xlim(500,1750)
    #plt.legend()
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_SiSpect_Log_NoFit.eps", format='eps')
    plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_SiSpect_Log_NoFit.png", format='png')
    plt_fit.show()





main()