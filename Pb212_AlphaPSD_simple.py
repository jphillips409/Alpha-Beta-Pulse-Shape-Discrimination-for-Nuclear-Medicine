########################################################
    # Created on April 30, 2025
# @author: Johnathan Phillips
# @email: j.s.phillips@wustl.edu

# Purpose: To unpack and analyze data for the Pb-212 decay using pulse-shape analysis.
#          PSD is performed by the digitizer and is read out as a psd_parameter.
#          This is a simplified version of another code
#          This will not perform waveform analysis and doesn't show some steps

# Detector: Liquid scintillation using the CAEN 5730B digitizer

# Liquid Scintillants:
#   Regular Ultima Gold
#   Ultima Gold AB: Meant for alpha-beta discrimination
#   Ultima Gold F: Meant for organic samples and provides high resolution.
#                  Must be mixed with AB to run the Pb-212 samples.
########################################################

# Import necessary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
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


# Unpacks the CAEN data in csv format. Can choose to unpack all data or some using alldat
# Can also choose to import the waveforms (a lot of data) or just the first 6 columns
#   which contain the important information for each event
def Unpack(name, alldat, waves):
    if alldat is True and waves is True:
        data = np.genfromtxt(name, skip_header = True, delimiter = ';')
    elif alldat is False and waves is True:
        data = np.genfromtxt(name, skip_header = True, delimiter =';', max_rows = 50000)
    elif alldat is True and waves is False:
        data = np.genfromtxt(name, skip_header = True, delimiter =';', usecols = (0, 1, 2, 3, 4, 5))
    else:
        data = np.genfromtxt(name, skip_header = True, delimiter =';', max_rows = 10000, usecols = (0, 1, 2, 3, 4, 5))

    return data


# Functions to accept and output an energy array given PSD gates and the PSD array
# For cut A - straight lines
# return trace array indices so that you can look at those waveforms
def PSDcutA(low, high, Earr, PSDarr):
    Filtarr = []
    trarr = []

    # iterates over the PSD array and filters for the high and low gates
    for i in range(len(PSDarr)):
        #if PSDarr[i] >= low and PSDarr[i] <= high:
            #Filtarr.append(Earr[i])
            #trarr.append(i)

        # For a non-straight lower PSD gate
        #if PSDarr[i] >= -0.0001166667 * Earr[i] + 0.35 and PSDarr[i] >= low and PSDarr[i] <= high:
       #if PSDarr[i] >= -0.0001442 * Earr[i] + 0.375 and PSDarr[i] >= low and PSDarr[i] <= high:
            #Filtarr.append(Earr[i])
            #trarr.append(i)

        # For a non straight lower PSD gate, UG with cleared blood, 30 ns SG
        #if PSDarr[i] >= -0.00031666 * Earr[i] + 0.35 and PSDarr[i] >= low and PSDarr[i] <= high:
            #Filtarr.append(Earr[i])
            #trarr.append(i)

            # For a non straight lower PSD gate, UG with cleared blood, 24 ns SG
        #if PSDarr[i] >= -0.00028333333 * Earr[i] + 0.4 and PSDarr[i] >= low and PSDarr[i] <= high:
            #Filtarr.append(Earr[i])
            #trarr.append(i)

            # For a non straight lower PSD gate, UG with mouse urine, 24 ns SG
        if PSDarr[i] >= -0.00028333333 * Earr[i] + high and PSDarr[i] >= low and PSDarr[i] <= high:
            Filtarr.append(Earr[i])
            trarr.append(i)

    return Filtarr, trarr

# For cut B - negatively sloped line
# return trace array indices so that you can look at those waveforms
def PSDcutB(Earr, PSDarr, psdval, UG):
    Filtarr = []
    trarr = []

    # iterates over the PSD array and filters for sloped gate
    for i in range(len(PSDarr)):
        # For regular Ultima Gold
        if PSDarr[i] >= -0.00021973 * Earr[i] + 1 and PSDarr[i] >= psdval and (UG == 'R' or UG == 'AB'):
            Filtarr.append(Earr[i])
            trarr.append(i)
        # For Ultima Gold AB+F with a coarse gain of 2.5 fC/(lsb x Vpp)
        #if PSDarr[i] >= -0.0002195685673 * Earr[i] + 1.32935 and PSDarr[i] >= psdval and PSDarr[i] <= 1 and UG == 'F':
         #   Filtarr.append(Earr[i])
          #  trarr.append(i)
        # For Ultima Gold AB+F with a coarse gain of 10 fC/(lsb x Vpp)
        #if PSDarr[i] >= -0.0006 * Earr[i] + 1 and PSDarr[i] >= psdval - 0.02:
            #Filtarr.append(Earr[i])
            #trarr.append(i)

        # Optional version where you are below the mixed line and above 700 ADC - for debugging
        # For Ultima Gold AB+F with a coarse gain of 2.5 fC/(lsb x Vpp)
        if PSDarr[i] <= -0.0002195685673 * Earr[i] + 1.32935 and PSDarr[i] >= psdval and PSDarr[i] <= 1 and UG == 'F' and Earr[i] >= 700:
            Filtarr.append(Earr[i])
            trarr.append(i)

    return Filtarr, trarr

# Make an energy cut for a psd region
# return trace array indices so that you can look at those waveforms
def Ecut(Earr, PSDarr, Eval, low, high):
    Filtarr = []
    trarr = []

    # iterates over the PSD array and filters for sloped gate
    for i in range(len(Earr)):
        if Earr[i] >= Eval and PSDarr[i] >= low and PSDarr[i] <= high:
            Filtarr.append(Earr[i])
            trarr.append(i)

    return Filtarr, trarr


# Plotting subroutines to clean up code

# Plotting a single 1D hist
def OneDHist(xdata,nbins,rangex,xlim,ylim,xtitle,ytitle):

    fig_oned, ax_oned = plt.subplots(layout='constrained')
    ax_oned.hist(xdata, bins=nbins, range=rangex)
    ax_oned.set_xlabel(xtitle, fontsize=20)
    ax_oned.set_ylabel(ytitle, fontsize=20)
    plt.ylim([0, ylim])
    plt.xlim([0, xlim])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # Optional for saving
    #if wavesdat is False:
        #plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_RawE_NoCosmic.eps",
                   # format='eps')
        #plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_RawE_NoCosmic.png",
                   # format='png')
    plt.show()

    return

# Plotting a single 2D hist
def TwoDHist(xdata,ydata,nbinsx,nbinsy,rangex,rangey,xlim,ylim,xtitle,ytitle):

    fig_twod, ax_twod = plt.subplots(layout = 'constrained')
    h = ax_twod.hist2d(xdata, ydata, bins=[nbinsx,nbinsy], range=[rangex, rangey], norm=mpl.colors.LogNorm(), cmin = 1)
    fig_twod.colorbar(h[3],ax=ax_twod)
    ax_twod.set_xlabel(xtitle, fontsize=20)
    ax_twod.set_ylabel(ytitle, fontsize=20)
    plt.ylim([0, ylim])
    plt.xlim([0, xlim])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # Optional for saving
    #if wavesdat is False:
        #plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_RawE_NoCosmic.eps",
                   # format='eps')
        #plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_RawE_NoCosmic.png",
                   # format='png')
    plt.show()

    return

# Function that accepts filtered energy data and returns a double Gaussian fit
# Parameters go as amplitude, position, standard deviation
# bl is a parameter for a constant baseline noise
#def DoubleGaussFit(x, a1, mu1, s1, a2, mu2, s2, bl):

    #return a1 * np.exp(-((x - mu1)**2)/(2 * s1**2)) + a2 * np.exp(-((x - mu2)**2)/(2 * s2**2)) + bl


def GaussFit(x, a, mu, s):

    return a * np.exp(-((x - mu)**2)/(2 * s**2))

def DoubleGaussFit(x, a, mu, s, a1, mu1, s1):

    return GaussFit(x, a, mu, s) + GaussFit(x, a1, mu1, s1)

def QuadGaussFit(x, a1, mu1, s1, a2, mu2, s2, a3, mu3, s3, a4, mu4, s4):

    return GaussFit(x, a1, mu1, s1) + GaussFit(x, a2, mu2, s2) + GaussFit(x, a3, mu3, s3) + GaussFit(x, a4, mu4, s4)

def FiveGaussFit(x, a1, mu1, s1, a2, mu2, s2, a3, mu3, s3, a4, mu4, s4, a5, mu5, s5):

    return GaussFit(x, a1, mu1, s1) + GaussFit(x, a2, mu2, s2) + GaussFit(x, a3, mu3, s3) + GaussFit(x, a4, mu4, s4) + GaussFit(x, a5, mu5, s5)

def SixGaussFit(x, a1, mu1, s1, a2, mu2, s2, a3, mu3, s3, a4, mu4, s4, a5, mu5, s5, a6, mu6, s6):

    return GaussFit(x, a1, mu1, s1) + GaussFit(x, a2, mu2, s2) + GaussFit(x, a3, mu3, s3) + GaussFit(x, a4, mu4, s4) + GaussFit(x, a5, mu5, s5) + GaussFit(x, a6, mu6, s6)

def SevenGaussFit(x, a1, mu1, s1, a2, mu2, s2, a3, mu3, s3, a4, mu4, s4, a5, mu5, s5, a6, mu6, s6, a7, mu7, s7):

    return GaussFit(x, a1, mu1, s1) + GaussFit(x, a2, mu2, s2) + GaussFit(x, a3, mu3, s3) + GaussFit(x, a4, mu4, s4) + GaussFit(x, a5, mu5, s5) + GaussFit(x, a6, mu6, s6) + GaussFit(x, a7, mu7, s7)


def main():
    # Read the CAEN data file in csv format
    # Different files available to process
    ###################################################

    # Remember to change your file path for your computer
    # Currently you must manually switch different PSD cut and peak fit settings for
    #   the different types of Ultima Gold scintillant.
    # Types of UG implemented:
    #       Regular UG, not mentioned in file name
    #       AB UG, denoted by "UGab" in filename
    #       AB + F UG, denoted by "UGf" in filename

    ###################################################
    # Constant filepath variable to get around the problem of backslashes in windows
    # The Path library will use forward slashes but convert them to correctly treat your OS
    # Also makes it easier to switch to a different computer
    #filepath = Path(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Pb212")

    #data_file = r'SDataR_WF_Pb212_Ar_2024_05_07_t60s.csv'
    #data_file =  r'SDataR_WF_Pb212_Ar_2024_05_07_t600.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_05_08a_t600.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_05_09a_t1800_10nsPG.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_05_09a_t2700.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_05_10a_t7200_10nsPG.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_05_11a_t14400_10nsPG.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_05_12a_t14400_10nsPG.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_05_13a_t28800_10nsPG.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_05_14a_t28800_10nsPG.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_05_15a_t28800_10nsPG.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_05_16a_t28800_10nsPG_400nsLG.csv'

    # Files that include a veto for cosmic ray events, now have a channel number
    #data_file = r'DataR_CH1@DT5730B_722_WF_Pb212_Ar_2024_05_17a_t600_10nsPG_410nsLG_NoVeto.csv'

    # Files that contain two channels (0, 1) but are a single file
    # These are set up so that:
    #   cosmic veto paddle: chan 0
    #   LSC:                chan 1
    #data_file = r'SDataR_WF_Pb212_Ar_2024_05_20a_t3600_10nsPG_410nsLG_SingleFile.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_05_20a_t7200_10nsPG_410nsLG_100lsbCR_SingleFile.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_05_21a_t14400_50lsbCR_SingleFile.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_05_21a_t11520_100lsbCR_SingleFile.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_05_22a_t10800_100lsbCR_SingleFile.csv'

    # Using the AB and F Ultima Gold
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_03a_t600_UGab.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_03a_t60_UGf.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_03a_t60_UGab.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_03a_t300_UGf_410LG_400TH_10CG.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_03a_t60_UGf_410LG_1000TH_10CG.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_03a_t60_UGf_410LG_400TH_10CG.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_03a_t60_UGf_410LG_448TH_10CG.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_04a_t600_UGab.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_04a_t600_UGf.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_04a_t600_UGf_RejectSat.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_04a_t1200_UGab.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_04a_t1200_UGf.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_05a_t2400_UGab.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_05a_t2400_UGf.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_05a_t3600_UGab.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_05a_t3600_UGf.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_06a_t3600_UGf_10CG.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_06a_t3600_UGab.csv'
   # data_file = r'DataR_WF_Pb212_Ar_2024_06_06a_t3600_UGf.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_06a_t2700_UGab.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_06a_t2700_UGf.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_07a_t7200_UGab.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_07a_t7200_UGf.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_08a_t14400_UGab.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_08a_t14400_UGf.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_09a_t14400_UGf.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_09a_t14400_Uab.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_10a_t14400_Uf.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_10a_t14400_Uab.csv'
    #data_file = r'DataR_WF_Pb212_Ar_2024_06_14a_t14400_Uf.csv'

    # Using the AB and F Ultima Gold with optical grease
    #data_file = r'SDataR_WF_Pb212_Ar_2024_07_09a_t300_Uf.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_07_10a_t600_Uf.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_07_11a_t3600_Uf.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_07_11a_t5400_Uf.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_07_12a_t14400_Uf.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_07_13a_t28800_Uf.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_07_14a_t28800_Uf.csv'


    # Blood Samples after this
    # Cleared blood sample with regular UG
    #data_file = r'SDataR_WF_Pb212_Ar_2024_010_16a_t240_UG_CBlood.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_010_16a_t240_UG_Blood.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_010_17a_t480_UG_CBlood.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_10_18b_t1200_UG_CBlood.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_10_19a_t10800_UG_CBlood.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_10_20a_t18000_UG_CBlood.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_10_21a_t21600_UG_CBlood.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_10_22a_t28800_UG_CBlood.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_10_23a_t28800_UG_CBlood.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_10_24a_t28800_UG_CBlood.csv'


    # Cleared blood sample with UGf
    #data_file = r'SDataR_WF_Pb212_Ar_2024_010_16a_t240_UGf_CBlood.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_010_17a_t480_UGf_CBlood.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2024_10_18a_t1200_UGf_CBlood.csv'


    # Mice urine 1.5 hrs post injection - 04/23/2025
    filepath = Path(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Pb212Mice\Ap23_2025_1_5hrs_postinjection")
    #data_file = r'SDataR_WF_Pb212_Ar_2025_04_23_t480_UG_Urine.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2025_04_23b_t480_UG_Urine.csv'
    data_file = r'SDataR_WF_Pb212_Ar_2025_04_23c_t480_UG_Urine.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2025_04_23d_t480_UG_Urine.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2025_04_23e_t480_UG_Urine.csv'

    # Mice urine 24 hrs post injection - 04/24/2025
    #filepath = Path(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Pb212Mice\Ap24_2025_24hrs_postinjection")
    #data_file = r'SDataR_WF_Pb212_Ar_2025_04_24_t60_UG_Urine.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2025_04_24a_t3600_UG_Urine.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2025_04_24b_t3600_UG_Urine.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2025_04_24c_t3600_UG_Urine.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2025_04_24d_t3600_UG_Urine.csv'


    # Mice urine 48 hrs post injection - 04/25/2025
    #filepath = Path(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Pb212Mice\Ap25_2025_48hrs_postinjection")
    #data_file = r'SDataR_WF_Pb212_Ar_2025_04_25_t60_UG_Urine.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2025_04_25a_t3600_UG_Urine.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2025_04_25b_t3600_UG_Urine.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2025_04_25c_t3600_UG_Urine.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2025_04_25d_t3600_UG_Urine.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2025_04_25e_t3600_UG_Urine.csv'


    # Mice urine w/ free Pb212 1 hr post injection - 05/14/2025
    #filepath = Path(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Pb212Mice\May14_2025_1hr_postinjection_nochelator")
    #data_file = r'SDataR_WF_Pb212_Ar_2025_05_14_t60_UG_Urine.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2025_05_14a_t1500_UG_Urine.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2025_05_14b_t1500_UG_Urine.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2025_05_14c_t1500_UG_Urine.csv'

    # Mice urine w/ free Pb212 24 hrs post injection - 05/15/2025
    # Cleaned excess grease from inside the LSC
    #filepath = Path(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Pb212Mice\May15_2025_24hrs_postinjection_nochelator")
    #data_file = r'SDataR_WF_Pb212_Ar_2025_05_15a_t3600_UG_Urine.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2025_05_15b_t3600_UG_Urine.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2025_05_15c_t3600_UG_Urine.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2025_05_15d_t3600_UG_Urine.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2025_05_15e_t3600_UG_Urine.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2025_05_15f_t3600_UG_Urine.csv'

    # Mice urine w/ free Pb212 48 hrs post injection - 05/16/2025
    #filepath = Path(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Pb212Mice\May16_2025_48hrs_postinjection_nochelator")
    #data_file = r'SDataR_WF_Pb212_Ar_2025_05_16a_t3600_UG_Urine.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2025_05_16b_t3600_UG_Urine.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2025_05_16c_t3600_UG_Urine.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2025_05_16d_t3600_UG_Urine.csv'
    #data_file = r'SDataR_WF_Pb212_Ar_2025_05_16e_t3600_UG_Urine.csv'


    # Now joins the path and file
    file_to_open = filepath / data_file

    # Reads the data file string and extracts run time
    # Looks for the undercase t and underscores - this must always be in the filename or this won't work
    # Reads in the string values, adds as strings, then converts to int
    data_string = list(data_file)
    runt_str = []
    runtime = ''
    for i in range(len(data_string)):
        if i != 0 and data_string[i] == 't' and data_string[i-1] == '_':
            for j in range(1,100):
                if data_string[i+j] == '_' or data_string[i+j] == '.': break
                else: runt_str.append(data_string[i+j])
        else: continue

    for i in range(len(runt_str)):
        runtime += runt_str[i]
    runtime = int(runtime)

    # Unpacks the data, choose alldat to True if you want to read all the file
    # Keep wavesdat at False
    # Truncated version is set to 10000 events
    alldat = True
    wavesdat = False
    data = Unpack(file_to_open, alldat, wavesdat)

    # Choose what type of Ultima Gold you are using
    #   R = Regular, AB, F = AB+F
    UG = 'R'
    #UG = 'AB'
    #UG = 'F'

    # Array to check event type
    chantype = []

    # The arrays used to store all the event information
    energy = []
    energy_short = []
    time = []
    channel = []
    psd_parameter = []
    traces = []

    satevents = 0 #  Counts the number of saturated events

    # The arrays used to store pairs of events across channels
    # Allows you to make a delta-T distribution and cut out cosmic rays
    time_pairs = []
    channel_pairs = []
    index_pairs = []
    dT = []
    coinc_wind = 1000000 # broad time window to ignore non-correlated events, 1 microsecond in picoseconds
    t_cuthigh = 7000 # actual time cut in picoseconds
    t_cutlow = -1000

    # Arrays for cosmic ray events
    # Only care about events in detector 1 - the LSC
    energy_cosmic = []
    psd_parameter_cosmic = []
    index_cosmic = []

    # Arrays that have had cosmic rays removed
    # Only care about channel 1
    energy_nocosmic = []
    channel_nocosmic = []
    time_nocosmic = []
    energy_short_nocosmic = []
    psd_parameter_nocosmic = []
    traces_nocosmic = []

    # Grab the channel number so you can determine the experimental setup
    # If only channel 0, earlier run, no cosmic paddle
    # If it contains channel 1, has cosmic paddle
    for i in range(len(data)):
        chantype.append(data[i][1])

    if 1 in chantype:
        print('Data contains paddle to detect cosmic rays: chan 0, LSC is chan 1')
    else:
        print('Data does not contain paddle to detect cosmic rays, LSC is chan 0')
    print('Runtime: ', runtime, 's')

    # Now fill in the data
    # Check event type before the loop, don't want to loop the event type check again and again
    if 1 in chantype:
        for i in range(len(data)):
            if data[i][3] > 0 and data[i][4] > 0:
                psd = (data[i][3]-data[i][4])/data[i][3]
                if (psd < 0 and data[i][1] == 1) or (psd > 1 and data[i][1] == 1): continue # ignores events with bad psd
                channel.append(data[i][1])
                time.append(data[i][2])
                energy.append(data[i][3])
                if data[i][3] == 4095 and data[i][1] == 1: satevents +=1
                energy_short.append(data[i][4])
                psd_parameter.append(psd)  # = (energy-energy_short)/energy
                if wavesdat is True: traces.append(data[i][6:])
    else:
        for i in range(len(data)):
            if data[i][3] > 0 and data[i][4] > 0:
                psd = (data[i][3]-data[i][4])/data[i][3]
                if psd < 0 or psd > 1: continue # ignores events with bad psd
                channel.append(data[i][1])
                time.append(data[i][2])
                energy.append(data[i][3])
                if data[i][3] == 4095 and data[i][1] == 1: satevents +=1
                energy_short.append(data[i][4])
                psd_parameter.append(psd)  # = (energy-energy_short)/energy
                if wavesdat is True: traces.append(data[i][6:])

    # Convert the data into numpy arrays
    energy = np.array(energy)
    energy_short = np.array(energy_short)
    psd_parameter = np.array(psd_parameter)
    time = np.array(time)
    channel = np.array(channel)
    traces = np.array(traces)
    psd_err = 0

    # Gets the difference between the first and last timestamp
    diff_t = time[-1] - time[0]
    print("1st and last dT is: ", diff_t, "ps")

    #Checks how many events have an unrealistic PSD parameter
    for i in range(len(psd_parameter)):
        if psd_parameter[i] < 0 or psd_parameter[i] > 1:
            #print(energy[i], "  ", psd_parameter[i])
            psd_err = psd_err + 1
    print("PSD too low or too high: ", psd_err)
    print("Saturated Events: ", satevents)

    # Finds the paired events within a microsecond window
    # Only used if the file contains two channels
    # Stores the of the event so you can find the energy and psd_parameter
    # Always sorted as [ channel 0 , channel 1 ]
    for i in range(len(channel) - 1):
        if np.abs(time[i+1] - time[i]) <= coinc_wind and channel[i+1] != channel[i]:
            if channel[i] == 0:
                channel_pairs.append([channel[i], channel[i+1]])
                index_pairs.append([i, i+1])
                time_pairs.append([time[i], time[i+1]])
            else:
                channel_pairs.append([channel[i+1], channel[i]])
                index_pairs.append([i+1, i])
                time_pairs.append([time[i+1], time[i]])

    for i in range(len(channel_pairs)):
        dT.append(time_pairs[i][0] - time_pairs[i][1])

    time_pairs = np.array(time_pairs)
    channel_pairs = np.array(channel_pairs)
    index_pairs = np.array(index_pairs)
    dT = np.array(dT)

    print(r'The number of time pairs in 1 $\mu$s is: ', len(time_pairs))

    # If pairs are found, plot the delta-T distribution and make a time cut
    if len(channel_pairs) > 0:
        dTf, dTax = plt.subplots(layout = 'constrained')
        dTax.set_title(r'$\Delta$T Distribution')
        dTax.set_ylabel('Counts')
        dTax.set_xlabel(r'Time (ps)')

        #Set some reasonable time window, 2 microseconds
        tStart = -25000
        tEnd = 25000
        dTax.set_xlim([tStart, tEnd])
        dTax.hist(dT, bins=1000, range = [tStart, tEnd])
        plt.show()

        # Make a time cut here
        # Want the value from channel 1
        for i in range(len(time_pairs)):
            if dT[i] >= t_cutlow and dT[i] <= t_cuthigh:
                index_cosmic.append(index_pairs[i][1])
                energy_cosmic.append(energy[index_pairs[i][1]])
                psd_parameter_cosmic.append(psd_parameter[index_pairs[i][1]])

    energy_cosmic = np.array(energy_cosmic)
    psd_parameter_cosmic = np.array(psd_parameter_cosmic)

    # Now go through the original lists and remove cosmic ray events and only use channel 1
    # Like the original data unpacking, don't want to loop over the event type check
    if 1 in channel:
        for i in range(len(channel)):
        # Need to be careful again here, earlier files only have channel 0 - LSC
            if i not in index_cosmic and channel[i] == 1:
                energy_nocosmic.append(energy[i])
                channel_nocosmic.append(channel[i])
                time_nocosmic.append(time[i])
                energy_short_nocosmic.append(energy_short[i])
                psd_parameter_nocosmic.append(psd_parameter[i])
                if wavesdat is True: traces_nocosmic.append(traces[i])
    else:
        for i in range(len(channel)):
            energy_nocosmic.append(energy[i])
            channel_nocosmic.append(channel[i])
            time_nocosmic.append(time[i])
            energy_short_nocosmic.append(energy_short[i])
            psd_parameter_nocosmic.append(psd_parameter[i])
            if wavesdat is True: traces_nocosmic.append(traces[i])

    traces_nocosmic = np.array(traces_nocosmic)
    energy_nocosmic = np.array(energy_nocosmic)

    print('The number of cosmic ray events is: ', len(energy_cosmic))

    # Prints the number of events at 4095 E, this is where saturation events should fall
    energy_saturation = energy_nocosmic[energy_nocosmic == 4095]
    print('Num events at saturation E (4095 ADC): ', len(energy_saturation))


    # Now we start plotting the raw data

    # Specify a number of bins
    nbins = 512

    # Plots the raw energy after removing cosmic rays - only channel 1
    # Also remove saturated events, i.e. E = 4095
    psd_parameter_nocosmic = np.array(psd_parameter_nocosmic)
    energy_nocosmic_nosat = energy_nocosmic[energy_nocosmic < 4095]

    OneDHist(energy_nocosmic_nosat, nbins, [0,4095],3000,4095,'ADC Channel', 'Counts')


    # Plots the PSD parameter vs the energy after removing cosmic rays - only channel 1
    TwoDHist(energy_nocosmic, psd_parameter_nocosmic, nbins, 500, [0,4095], [0,1], 4095, 1, 'ADC Channel', 'PSD parameter')

    ############################################################################################################
    # Makes PSD cuts

    # Sets your A PSD gate - straight lines
    # Also returns an array containing the indices for the traces
    # Using regular Ultima Gold
    PSDlowA = 0.13
    PSDhighA = 0.25

    # Regular UG but with grease, cleared blood, and 24 ns short gate
    PSDlowA = 0.23
    PSDhighA = 0.4

    # Regular UG but with grease, mouse urine, and 24 ns short gate
    PSDlowA = 0.18
    PSDhighA = 0.35

    # Using AB Ultima Gold
    if UG == 'AB': PSDlowA = 0.27
    if UG == 'AB': PSDhighA = 0.45

    # Using Ultima Gold AB + F
    if UG == 'F': PSDlowA = 0.25
    if UG == 'F': PSDhighA = 0.43

    # If using UG AB + F and optical grease
    if UG == 'F': PSDlowA = 0.27
    if UG == 'F': PSDhighA = 0.4

    energy_filtA, tracesind_filtA = PSDcutA(PSDlowA, PSDhighA, energy_nocosmic, psd_parameter_nocosmic)
    print("Events that have a non-zero energyshort and energylong: ", len(energy_nocosmic))
    print("Events inside the PSD cut A: ", len(energy_filtA))

    energy_filtA = np.array(energy_filtA)

    #Get the times for these events
    time_filtA = []
    for i in range(len(tracesind_filtA)):
        time_filtA.append(time_nocosmic[tracesind_filtA[i]])
    time_filtA = np.array(time_filtA)

    # If using Ultima Gold F on a 10 fC/(lsb x Vpp) setting, get events between alpha peaks
    #energy_filtAbtw_temp = energy_filtA[energy_filtA >= 400]
    #energy_filtAbtw_temp = np.array(energy_filtAbtw_temp)
    #energy_filtAbtw = energy_filtAbtw_temp[energy_filtAbtw_temp <= 500]

    BiAlphaCPS = energy_filtA[energy_filtA >= 550]
    BiAlphaCPS = BiAlphaCPS[BiAlphaCPS <= 1050]
    #pltBi, axBi = plt.subplots(layout = 'constrained')
    #axBi.hist(BiAlphaCPS, bins=nbins, range=[0, 4095])
    #plt.xlim(0,2000)
    #plt.show()
    print("Counts in Bi alpha PSD/energy region: ", len(BiAlphaCPS))
    print("CPS in Bi alpha PSD/energy region: ", len(BiAlphaCPS)/runtime)

    # Sets your B PSD gate - negatively sloped to get the mixed beta/alpha
    # Also cut to not overlap with gate A
    # The line equation is stored in the corresponding function
    # Also returns an array containing the indices for the traces
    energy_filtB, tracesind_filtB = PSDcutB(energy_nocosmic, psd_parameter_nocosmic, PSDhighA, UG)
    print("Events inside the PSD cut B: ", len(energy_filtB))
    energy_filtB = np.array(energy_filtB)


    # Sets an energy gate for mixed beta/alpha that don't fall into the short gate
    # E cut set to 1200, psd cut set to PSDlowA
    # Also returns an array containing the indices for the traces
    # Using regular Ultima Gold
    PSDlowC = 0.0

    #Using Ultima Gold AB+F
    #PSDlowC = 0.1
    PSDhighC = PSDlowA

    energy_filtC, tracesind_filtC = Ecut(energy_nocosmic, psd_parameter_nocosmic, 150, PSDlowC, PSDhighC)
    print("Events inside the energy cut (C): ", len(energy_filtC))
    print("Rate of events inside cut C (/s): ", len(energy_filtC) / (runtime))  # Use the run time
    print("")
    energy_filtC = np.array(energy_filtC)


    # Plots the PSD parameter vs the energy with the cut
    # Draws the straight A cuts
    xline = np.linspace(0,4096,10)
    #yline = np.linspace(0,PSDlowA,10)
    #PSDlowlineA = np.full(len(xline), PSDlowA)
    PSDhighlineA = np.full(len(xline), PSDhighA)

    # For a non-straight A cut
    #xline = np.linspace(0, (PSDlowA - 0.35) / (-0.0001166667),9)
    #yline = np.full(len(xline), -0.0001166667 * xline + 0.35)
    #xline = np.linspace(0, (PSDlowA - 0.375) / (-0.0001442),9)
    #yline = np.full(len(xline), -0.0001442 * xline + 0.375)
    #xline = np.append(xline, 4096)
    #PSDlowlineA = np.append(yline, PSDlowA)

    # For a non-straight A cut - regular UG with clear blood - 30 ns gate
    #xline = np.linspace(0, (PSDlowA - 0.35) / (-0.0003166667),9)
    #yline = np.full(len(xline), -0.0003166667 * xline + 0.34)
    #xline = np.append(xline, 4096)
    #PSDlowlineA = np.append(yline, PSDlowA)

    # For a non-straight A cut - regular UG with clear blood - 24 ns gate
    #xline = np.linspace(0, (PSDlowA - 0.4) / (-0.000283333),9)
    #yline = np.full(len(xline), -0.000283333 * xline + 0.4)
    #xline = np.append(xline, 4096)
    #PSDlowlineA = np.append(yline, PSDlowA)

    # For a non-straight A cut - regular UG with mouse urine - 24 ns gate
    xline = np.linspace(0, (PSDlowA - PSDhighA) / (-0.000283333),9)
    yline = np.full(len(xline), -0.000283333 * xline + PSDhighA)
    xline = np.append(xline, 4096)
    PSDlowlineA = np.append(yline, PSDlowA)

    #Draws the B cut
    # For regular Ultima Gold
    xlineB = np.linspace(0, (PSDhighA - 1) / (-0.00021973),9)
    PSDlineB = np.full(len(xlineB), -0.00021973 * xlineB + 1)
    xlineB = np.append(xlineB, 4096)
    PSDlineB = np.append(PSDlineB, PSDhighA)

    # For Ultima Gold AB+F with a coarse gain of 2.5 fC/(lsb x Vpp)
    #if UG == 'F':
        #xlineB = np.linspace(1500, 4096,10)
        #PSDlineB = np.full(len(xlineB), -0.0002195685673 * xlineB + 1.32935)

    # For Ultima Gold AB+F with a coarse gain of 10 fC/(lsb x Vpp)
    #xlineB = np.linspace(0, (PSDhighA - 0.02 - 1) / (-0.0006),9)
    #PSDlineB = np.full(len(xlineB), -0.0006 * xlineB + 1)
    #xlineB = np.append(xlineB, 4096)
    #PSDlineB = np.append(PSDlineB, PSDhighA - 0.02)

    ElineC = np.full(len(yline), 400)

    fig_psdE2, ax_psdE2 = plt.subplots(layout = 'constrained')
    h2 = ax_psdE2.hist2d(energy_nocosmic, psd_parameter_nocosmic, bins=[nbins,500], range=[[0,4095], [0,1]], norm=mpl.colors.LogNorm(), cmin = 1)
    fig_psdE2.colorbar(h2[3],ax=ax_psdE2)
    ax_psdE2.plot(xline, PSDlowlineA, color='black', linewidth = 3)
    ax_psdE2.plot(xline, PSDhighlineA, color='black', linewidth = 3)
    ax_psdE2.plot(xlineB, PSDlineB, color='red', linewidth = 3, zorder = 10)
    #ax_psdE2.plot(ElineC, yline, color='blue', linewidth = 3 )
    plt.ylim([0, 1])
    plt.xlim([0,4095])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    ax_psdE2.set_ylabel('PSD parameter',fontsize=20)
    ax_psdE2.set_xlabel('ADC Channel',fontsize=20)
    if wavesdat is False:
        plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_PSDvsE_Gates.eps", format='eps')
        plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_PSDvsE_Gates.png", format='png')
    plt.show()


    # Plots the filtered energy data for cut A
    fig_filtEA, ax_filtEA = plt.subplots(layout = 'constrained')
    (data_entries, bins, patches) = ax_filtEA.hist(energy_filtA, bins=nbins, range = [0,4095])
    ax_filtEA.set_xlabel('ADC Channel',fontsize=20)
    ax_filtEA.set_ylabel('Counts',fontsize=20)
   # ax_filtEA.set_yscale('log')
    plt.xlim(0,3500) # With UG and Cleared Blood
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim([0,4500])
    #plt.xlim(0,4095)
    if wavesdat is False:
        plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_AlphaGate.eps", format='eps')
        plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_AlphaGate.png", format='png')
    plt.show()

    # Plots the filtered energy data for cut B
    fig_filtEB, ax_filtEB = plt.subplots(layout = 'constrained')
    (data_entriesB, binsB, patchesB) = ax_filtEB.hist(energy_filtB, bins=nbins, range = [0,4095])
    ax_filtEB.set_xlabel('ADC Channel')
    ax_filtEB.set_ylabel('Counts')
    plt.xlim([0., 4096.])

    plt.show()

    # Plots the filtered energy data for cut C
    fig_filtEC, ax_filtEC = plt.subplots(layout = 'constrained')
    (data_entriesC, binsC, patchesC) = ax_filtEC.hist(energy_filtC, bins=nbins, range = [0,4095],label='cutC')
    ax_filtEC.set_xlabel('ADC Channel')
    ax_filtEC.set_ylabel('Counts')

    plt.show()

    # Array that holds all values within the PSD cuts
    energy_filtTot = np.concatenate((energy_filtA, energy_filtB, energy_filtC))

    # Plots the filtered energy for cuts A + B + C to compare with total
    fig_filtEtot, ax_filtEtot = plt.subplots(layout = 'constrained')
    (data_entriestot, binstot, patchestot) = ax_filtEtot.hist(energy_filtTot, bins=nbins, range = [0,4095])
    ax_filtEtot.set_xlabel('ADC Channel')
    ax_filtEtot.set_ylabel('Counts')

    plt.show()

    # Prints out the backgrounds
    print("Total background: ", len(energy_nocosmic)/runtime, " CPS")
    print("Total beta background: ", len(energy_filtC)/runtime, " CPS")
    EC_bool = np.asarray(energy_filtC)
    print("Beta background above 100 ADC Chans: ", (EC_bool > 100).sum()/runtime, " CPS")
    print("Beta background above 200 ADC Chans: ", (EC_bool > 200).sum()/runtime, " CPS")
    print("")


    ############################################################################################################
    # Stacks two plots on top of each other
    #   PSD vs E with gates drawn
    #   Spectra from region A
    #Plots fit and 2d vertically together
    fig = plt.figure(layout='constrained',figsize=(8,12))
    gateAax_stack = fig.add_subplot(2,1,2)
    PSDvsEax_stack = fig.add_subplot(2,1,1)
    gateAax_stack.hist(energy_nocosmic, bins=nbins, range=[0, 4095])
    #gateAax_stack.legend()
    gateAax_stack.set_xlim(0,4095)
    gateAax_stack.set_xlabel('ADC Channel',fontsize=24)
    gateAax_stack.set_ylabel('Counts',fontsize=24)

    h2 = PSDvsEax_stack.hist2d(energy_nocosmic, psd_parameter_nocosmic, bins=[nbins,500], range=[[0,4095], [0,1]], norm=mpl.colors.LogNorm(), cmin = 1, rasterized=True)
    fig.colorbar(h2[3],ax=PSDvsEax_stack,fraction=0.046, pad=0.04)
    #PSDvsEax_stack.plot(xline, PSDlowlineA, color='black', linewidth = 3)
    #PSDvsEax_stack.plot(xline, PSDhighlineA, color='black', linewidth = 3)
    #PSDvsEax_stack.plot(xlineB, PSDlineB, color='red', linewidth = 3, zorder = 10)
    plt.ylim([0, 1])
    plt.xlim([0,4095])
    PSDvsEax_stack.set_ylabel('PSD Parameter',fontsize=24)
    #plt.xticks(fontsize=15)
    #plt.yticks(fontsize=15)

    if wavesdat is False:
        plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_Stacked.eps", format='eps')
        plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_Stacked.png", format='png')
        plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_Stacked.svg", format='svg')

    plt.show()

    plt.show()



    ############################################################################################################
    # Fitting the alpha peaks

    # Get bin centers
    # Get bin centers
    bincenters = np.array([0.5 * (bins[i] + bins[i + 1])  for i in range(len(bins) - 1)])

    # Find the bin width
    binwidth = bins[1] - bins[0]

    # Define range for each peak
    # For regular Ultima Gold
    lower_bound1 = 500
    upper_bound1 = 1100

    lower_bound2 = 1000
    upper_bound2 = 1800

    # For regular Ultima Gold with optical grease and for mouse urine
    # Might vary from sample to sample, probably due to mouse hydration
    lower_bound1 = 700
    upper_bound1 = 1400

    lower_bound2 = 1500
    upper_bound2 = 2400

    # For Ultima Gold AB
    if UG == 'AB':
        lower_bound1 = 700
        upper_bound1 = 1400

        lower_bound2 = 1400
        upper_bound2 = 2200

    # For Ultima Gold AB+F with a coarse gain of 2.5 fC/(lsb x Vpp)
    if UG == 'F':
        lower_bound1 = 1000
        upper_bound1 = 1700

        lower_bound2 = 2000
        upper_bound2 = 2900

        # With optical grease
        lower_bound1 = 1400
        upper_bound1 = 2400

        lower_bound2 = 2800
        upper_bound2 = 3800

    # For Ultima Gold AB+F with a coarse gain of 10 fC/(lsb x Vpp)
    #lower_bound1 = 200
    #upper_bound1 = 450

    #lower_bound2 = 450
    #upper_bound2 = 750


    # Get points for each bin center
    xpeak1 = bincenters[bincenters>lower_bound1]
    x1 = xpeak1[xpeak1<upper_bound1]
    x1 = x1.ravel()

    y1 = data_entries[bincenters>lower_bound1]
    y1 = y1[xpeak1<upper_bound1]
    y1 = y1.ravel()

    xpeak2 = bincenters[bincenters>lower_bound2]
    x2 = xpeak2[xpeak2<upper_bound2]
    x2 = x2.ravel()

    y2 = data_entries[bincenters>lower_bound2]
    y2 = y2[xpeak2<upper_bound2]
    y2 = y2.ravel()

    plotx1, axx1 = plt.subplots(layout = 'constrained')
    axx1.plot(x1,y1)
    axx1.set_xlabel('ADC Channel')
    axx1.set_ylabel('Counts')
    plt.show()

    plotx2, axx2 = plt.subplots(layout = 'constrained')
    axx2.plot(x2,y2)
    plt.show()

    # Set parameter guesses
    # For regular Ultima Gold
    p01 = ([1000, 900, 200])
    p02 = ([1000, 1700, 100])

    # For Ultima Gold AB
    if UG == 'AB':
        p01 = ([2000, 900, 200])
        p02 = ([2000, 1700, 100])

    # For Ultima Gold AB + F with a coarse gain of 2.5 fC/(lsb x Vpp)
    if UG == 'F':
        p01 = ([2000, 1300, 200])
        p02 = ([2000, 2400, 200])

        # With optical grease
        p01 = ([2000, 1800, 200])
        p02 = ([2000, 3400, 200])

    # For Ultima Gold AB+F with a coarse gain of 10 fC/(lsb x Vpp)
    #p02 = ([2000, 600, 100])    #p01 = ([2000, 325, 200])


    # Fits the data to 2 Gaussians and plots
    E_param1, E_cov1 = curve_fit(GaussFit, xdata = x1, ydata = y1, p0 = p01, maxfev = 10000)
    E_param2, E_cov2 = curve_fit(GaussFit, xdata = x2, ydata = y2, p0 = p02, maxfev = 10000)
    E_err1 = np.sqrt(np.diag(E_cov1))
    E_err2 = np.sqrt(np.diag(E_cov2))


    xspace1 = np.linspace(lower_bound1, upper_bound1, 1000)
    xspace2 = np.linspace(lower_bound2, upper_bound2, 1000)

    # Plots the hitogram and fitted function
    fitplt, fitax = plt.subplots(layout = 'constrained')
    fitax.hist(energy_filtA, bins=nbins, range = [0,4095], label = r'$\alpha$ Events')
    fitax.plot(xspace1, GaussFit(xspace1, *E_param1), linewidth = 2.5, label = r'$^{212}$Bi $\alpha$ fit')
    fitax.plot(xspace2, GaussFit(xspace2, *E_param2), linewidth = 2.5, label = r'$^{212}$Po $\alpha$ fit')
    plt.legend(loc='upper right',fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(0,4095) # With UG and Cleared Blood
    plt.ylim(0,2500) # With F UG
    #plt.xlim(0,4095) # With F UG and optical grease
    #plt.ylim(0,20)
    fitax.set_xlabel('ADC Channel',fontsize=20)
    fitax.set_ylabel('Counts',fontsize=20)
    if wavesdat is False:
        plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_Fit.eps", format='eps')
        plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_Fit.png", format='png')
        plt.savefig(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Paper_Figures\Pb212_Fit.svg", format='svg')

    plt.show()

    # Calculates the integral
    GInt1, GIntErr1 = quad(GaussFit,lower_bound1, upper_bound1, args=(E_param1[0], E_param1[1], E_param1[2]))
    GInt2, GIntErr2 = quad(GaussFit,lower_bound2, upper_bound2, args=(E_param2[0], E_param2[1], E_param2[2]))

    # Correct integrals for bin width
    GInt1 = GInt1/binwidth
    GInt2 = GInt2/binwidth

    # Use counting statistics for the uncertainty: ~ sqrt(num counts)
    GIntErr1 = np.sqrt(GInt1)
    GIntErr2 = np.sqrt(GInt2)

    # Prints out the fitting parameters and integrals
    print("The fit parameters are:")
    print("Amplitude 1: ", E_param1[0], " +/-", E_err1[0], " Counts")
    print("Mean 1: ", E_param1[1], " +/-", E_err1[1], " ADC Channel")
    print("Stdev 1: ", E_param1[2], " +/-", E_err1[2], " ADC Channel")
    print("Integral 1: ", GInt1, " +/-", GIntErr1, " Counts")
    print("Amplitude 2: ", E_param2[0], " +/-", E_err2[0], "Counts")
    print(" Mean 2: ", E_param2[1], " +/-", E_err2[1], "ADC Channel")
    print(" Stdev 2: ", E_param2[2], " +/-", E_err2[2], " ADC Channel")
    print("Integral 2: ", GInt2, " +/-", GIntErr2, " Counts")

    print("")
    print(r"Total $^{212}$Pb decays: ", GInt1 + GInt2 + len(energy_filtB), " +/- ", np.sqrt(GInt1 + GInt2 + len(energy_filtB)), " Counts")

    # Calculate the activity based on the run time
    # Could make it automatic, use timestamp[last] - timestamp[0] but that would be slightly off
    # Might be a good enough approx
    Pbactivity = (GInt1 + GInt2 + len(energy_filtB) + len(energy_filtC))/runtime  # CPS
    Pbactivity = Pbactivity * (1/37000) # microcurie
    # Use relative counting uncertainty to get activity uncertainty
    print(r"Total $^{212}$Pb activity: ", Pbactivity, " +/- ", np.sqrt(GInt1 + GInt2) * Pbactivity / (GInt1 + GInt2))

    # Calculates and prints the 212Po branching ratio
    print(r"The Pb-212 branching ratio is: ", (GInt2 + len(energy_filtB))/(GInt1 + GInt2 + len(energy_filtB)))

    #Adds up peak counts instead of integrating
    peak1_add = ((500 < energy_filtA) & (energy_filtA < 950)).sum()
    peak2_add = ((1000 < energy_filtA) & (energy_filtA < 1400)).sum()

    print("Bin sums for peaks 1: ", peak1_add, " and 2: ", peak2_add)

    #Adds up alpha at low energy, really only see these with UG F and optical grease
    first_second_decay = ((1400 < energy_filtA) & (energy_filtA < 2300)).sum()
    fourth_decay = ((550 < energy_filtA) & (energy_filtA < 950)).sum()
    third_decay = ((950 < energy_filtA) & (energy_filtA < 1400)).sum()
    print("Events at ~ 700 peak between 550 and 950 ", fourth_decay, " % of Bi-212 alpha decays: ", fourth_decay/first_second_decay)
    print("Events at ~ 1200 peak between 950 and 1400 ", third_decay, " % of Bi-212 alpha decays: ", third_decay/first_second_decay)


    # Finds the number of alpha at a specific time in the run
    # Start with the first 8th and the last 8th
    alphas_first8th = 0
    alphas_last8th = 0
    first8th = (1/8)*diff_t
    for i in range(len(time_filtA)):
        if energy_filtA[i] > 750 and energy_filtA[i] < 2500 and time_filtA[i] < first8th: alphas_first8th += 1

        if energy_filtA[i] > 750 and energy_filtA[i] < 2500 and time_filtA[i] > diff_t - first8th: alphas_last8th += 1

    print('Alphas in the first 8th: ', alphas_first8th)
    print('Alphas in the last 8th: ', alphas_last8th)


main()