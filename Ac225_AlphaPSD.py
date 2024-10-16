########################################################
# Created on May 11 2024
# @author: Johnathan Phillips
# @email: j.s.phillips@wustl.edu

# Purpose: To unpack and analyze data for the Ac-225 decay using pulse-shape analysis.
#          PSD is performed by the digitizer and is read out as a psd_parameter.

# Detector: Liquid scintillation using the CAEN 5730B digitizer

# Liquid Scintillants:
#   Regular Ultima Gold
#   Ultima Gold AB: Meant for alpha-beta discrimination
#   Ultima Gold F: Meant for organic samples and provides high resolution.
#                  Must be mixed with AB to run the Ac-225 samples.
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

        # For a non-straight lower PSD gate with UG F
        #if PSDarr[i] >= -0.0001166667 * Earr[i] + 0.35 and PSDarr[i] >= low and PSDarr[i] <= high:
        if PSDarr[i] >= -0.0001442 * Earr[i] + 0.375 and PSDarr[i] >= low and PSDarr[i] <= high:
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
        if PSDarr[i] <= -0.000184168 * Earr[i] + 1.18417 and PSDarr[i] >= psdval and PSDarr[i] <= 1 and UG == 'F' and Earr[i] >= 700:
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

# Function that accepts filtered energy data and returns a double Gaussian fit
# Parameters go as amplitude, position, standard deviation
# bl is a parameter for a constant baseline noise
def DoubleGaussFit(x, a1, mu1, s1, a2, mu2, s2, bl):

    return a1 * np.exp(-((x - mu1)**2)/(2 * s1**2)) + a2 * np.exp(-((x - mu2)**2)/(2 * s2**2)) + bl


def GaussFit(x, a, mu, s):

    return a * np.exp(-((x - mu)**2)/(2 * s**2))

def TripleGaussFit(x, a1, mu1, s1, a2, mu2, s2, a3, mu3, s3, bl):

    return GaussFit(x, a1, mu1, s1) + GaussFit(x, a2, mu2, s2) + GaussFit(x, a3, mu3, s3) + bl

def QuadGaussFit(x, a1, mu1, s1, a2, mu2, s2, a3, mu3, s3, a4, mu4, s4, bl):

    return GaussFit(x, a1, mu1, s1) + GaussFit(x, a2, mu2, s2) + GaussFit(x, a3, mu3, s3) + GaussFit(x, a4, mu4, s4) + bl

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
    # All or almost all Ac225 samples should be Ultima Gold AB + F

    ###################################################
    # Constant filepath variable to get around the problem of backslashes in windows
    # The Path library will use forward slashes but convert them to correctly treat your OS
    # Also makes it easier to switch to a different computer
    filepath = Path(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Ac225")

    #data_file = r'SDataR_DataR_WF_Ac225_Ar_2024_06_21a_t60_Uf.csv'
    #data_file = r'SDataR_DataR_WF_Ac225_Ar_2024_06_22a_t60_Uf.csv'
    #data_file = r'SDataR_DataR_WF_Ac225_Ar_2024_06_25a_t60_Uf.csv'
    #data_file = r'SDataR_DataR_WF_Ac225_Ar_2024_06_26a_t60_Uf.csv'
    #data_file = r'SDataR_DataR_WF_Ac225_Ar_2024_06_27a_t60_Uf.csv'
    #data_file = r'SDataR_DataR_WF_Ac225_Ar_2024_07_01a_t60_Uf.csv'
    #data_file = r'SDataR_DataR_WF_Ac225_Ar_2024_07_01b_t60_Uf.csv'
    data_file = r'SDataR_DataR_WF_Ac225_Ar_2024_07_12a_t240_Uf.csv'
    #data_file = r'SDataR_DataR_WF_Ac225_Ar_2024_08_02a_t720_Uf.csv'


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
    # Set wavesdat to True if you want the waveforms
    # Truncated version is set to 10000 events
    alldat = False
    wavesdat = True
    data = Unpack(file_to_open, alldat, wavesdat)

    # Choose what type of Ultima Gold you are using
    #   R = Regular, AB, F = AB+F
    #UG = 'R'
    #UG = 'AB'
    UG = 'F'

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
    print("1st and last dT is: ", time[len(time) - 1] - time[0], "ps")

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

    print('The number of cosmic ray events is: ', len(energy_cosmic))
    wavetime = 0
    # Plot traces.
    # Set up plot window.
    if wavesdat is True:
        f, ax = plt.subplots(layout = 'constrained')
        ax.set_title('Traces')
        ax.set_ylabel('Voltage')
        ax.set_xlabel('Time (ns)')

        xStart = 0
        xEnd = len(traces[1]) - 1
        wavetime = np.linspace(xStart, xEnd, len(traces[1]))

        # Convert sample number to time
        # CAEN samples every 2 ns
        wavetime = wavetime * 2

        for i in range(1): # Plot the first five neutron traces
            ax.plot(wavetime, traces[0], label=' trace')
        ax.set_xlim([xStart, xEnd * 2])
        ax.set_ylim([9000, 14000])

        #plt.legend()
        plt.show(block=True) # Don't block terminal by default.
        #quit()

    # Now we start plotting the raw data

    # Specify a number of bins
    nbins = 512

    # Plots the raw spectrum
    fig_rawE, ax_rawE = plt.subplots(layout = 'constrained')
    ax_rawE.hist(energy, bins=nbins, range = [0,4095])
    ax_rawE.set_title('Raw ADC Channel')
    ax_rawE.set_xlabel('ADC Channel')
    ax_rawE.set_ylabel('Counts')
    plt.show()

    # Plots the raw PSD parameter
    fig_PSD, ax_PSD = plt.subplots(layout = 'constrained')
    ax_PSD.hist(psd_parameter,bins=4000, range = [0,1])
    ax_PSD.set_title('Raw PSD Parameter')
    ax_PSD.set_xlabel('PSD Parameter')
    ax_PSD.set_ylabel('Counts')
    plt.xlim([0.,1.])
    plt.show()

    # Plots the raw PSD parameter vs the energy
    fig_psdE, ax_psdE = plt.subplots(layout = 'constrained')
    h = ax_psdE.hist2d(energy, psd_parameter, bins=[nbins,500], range=[[0,4095], [0,1]], norm=mpl.colors.Normalize(), cmin = 1)
    fig_psdE.colorbar(h[3],ax=ax_psdE)
    plt.ylim([0., 1.])
    ax_psdE.set_title('Raw Energy vs PSD Parameter')
    ax_psdE.set_ylabel('PSD Parameter')
    ax_psdE.set_xlabel('ADC Channel')
    plt.show()

    # Plots the E spectrum for cosmic ray events - only channel 1
    if len(energy_cosmic) > 0:
        # Grab only the events in the beta psd region
        Ecosmic = energy_cosmic[psd_parameter_cosmic <= 0.2]
        fig_Ecosmic, ax_Ecosmic = plt.subplots(layout = 'constrained')
        ax_Ecosmic.hist(Ecosmic, bins=250, range = [0,4095])
        ax_Ecosmic.set_title('Cosmic Ray Events')
        ax_Ecosmic.set_xlabel('ADC Channel')
        ax_Ecosmic.set_ylabel('Counts/16 Channels')
        plt.show()

        # Plots the PSD parameter vs the energy for cosmic ray events - only channel 1
        fig_psdE_cosmic, ax_psdE_cosmic = plt.subplots(layout = 'constrained')
        h = ax_psdE_cosmic.hist2d(energy_cosmic, psd_parameter_cosmic, bins=[100,100], range=[[0,4095], [0,1]], norm=mpl.colors.Normalize(), cmin = 1)
        fig_psdE_cosmic.colorbar(h[3],ax=ax_psdE_cosmic)
        plt.ylim([0., 1.])
        ax_psdE_cosmic.set_title('Energy vs PSD Parameter for Cosmic Rays')
        ax_psdE_cosmic.set_ylabel('PSD parameter')
        ax_psdE_cosmic.set_xlabel('ADC Channel')
        plt.show()

    # Plots the PSD parameter vs the energy before removing cosmic rays - only channel 1
    E1cosmic = energy[channel == 1]
    PSD1cosmic = psd_parameter[channel == 1]
    fig_psdE_withcosmic, ax_psdE_withcosmic = plt.subplots(layout = 'constrained')
    h = ax_psdE_withcosmic.hist2d(E1cosmic, PSD1cosmic, bins=[nbins,500], range=[[0,4095], [0,1]], norm=mpl.colors.Normalize(), cmin = 1)
    fig_psdE_withcosmic.colorbar(h[3],ax=ax_psdE_withcosmic)
    plt.ylim([0., 1.])
    ax_psdE_withcosmic.set_title('Energy vs PSD Parameter with Cosmic Rays')
    ax_psdE_withcosmic.set_ylabel('PSD parameter')
    ax_psdE_withcosmic.set_xlabel('ADC Channel')
    plt.show()

    # Plots the PSD parameter vs the energy after removing cosmic rays - only channel 1
    fig_psdE_nocosmic, ax_psdE_nocosmic = plt.subplots(layout = 'constrained')
    h = ax_psdE_nocosmic.hist2d(energy_nocosmic, psd_parameter_nocosmic, bins=[nbins,500], range=[[0,4095], [0,1]], norm=mpl.colors.Normalize(), cmin = 1)
    fig_psdE_nocosmic.colorbar(h[3],ax=ax_psdE_nocosmic)
    plt.ylim([0., 1.])
    #ax_psdE_nocosmic.set_title('Energy vs PSD Parameter without Cosmic Rays')
    ax_psdE_nocosmic.set_ylabel('PSD parameter')
    ax_psdE_nocosmic.set_xlabel('ADC Channel')
    plt.show()

    ############################################################################################################
    # Makes PSD cuts

    # Sets your A PSD gate - straight lines
    # Also returns an array containing the indices for the traces
    # Using regular Ultima Gold
    PSDlowA = 0.13
    PSDhighA = 0.25

    # Using AB Ultima Gold
    if UG == 'AB': PSDlowA = 0.27
    if UG == 'AB': PSDhighA = 0.45

    # Using Ultima Gold AB + F
    if UG == 'F': PSDlowA = 0.23 #0.25
    if UG == 'F': PSDhighA = 0.38 #0.43

    energy_filtA, tracesind_filtA = PSDcutA(PSDlowA, PSDhighA, energy_nocosmic, psd_parameter_nocosmic)
    print("Events that have a non-zero energyshort and energylong: ", len(energy_nocosmic))
    print("Events inside the PSD cut A: ", len(energy_filtA))

    energy_filtA = np.array(energy_filtA)

    # If using Ultima Gold F on a 10 fC/(lsb x Vpp) setting, get events between alpha peaks
    #energy_filtAbtw_temp = energy_filtA[energy_filtA >= 400]
    #energy_filtAbtw_temp = np.array(energy_filtAbtw_temp)
    #energy_filtAbtw = energy_filtAbtw_temp[energy_filtAbtw_temp <= 500]

    BiAlphaCPS = energy_filtA[energy_filtA >= 550]
    BiAlphaCPS = BiAlphaCPS[BiAlphaCPS <= 1050]
    pltBi, axBi = plt.subplots(layout = 'constrained')
    axBi.hist(BiAlphaCPS, bins=nbins, range=[0, 4095])
    plt.xlim(0,2000)
    plt.show()
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


    # Plots traces for each cut
    if wavesdat is True:
        ftr_A, axtr_A = plt.subplots(layout = 'constrained')
        # Looks for traces that fall within the gate and plot them
        #for i in range(10):
            #axtr_A.plot(wavetime, traces_nocosmic[tracesind_filtA[i]], label=' trace')
        axtr_A.plot(wavetime, traces_nocosmic[tracesind_filtA[4]], label=' trace')
       # axtr_A.set_title('Traces in Cut A')
        axtr_A.set_ylabel('Voltage (mV)')
        yStart = 8000
        yEnd = 14000
        axtr_A.set_xlabel('Time (ns)')
        axtr_A.set_xlim([75, 175])
        axtr_A.set_ylim([8000, yEnd])
        plt.show(block=True)  # Don't block terminal by default.

        ftr_Abtw, axtr_Abtw = plt.subplots(layout = 'constrained')
        # Looks for traces that fall within the the two alpha peaks and plot them
        incr = 0
        #for i in range(1500):
            #if energy_filtA[i] >= 300 and energy_filtA[i] <= 400 and i > 900: # Peak 1
                #axtr_Abtw.plot(wavetime, traces_nocosmic[tracesind_filtA[i]], label=' trace')
            #if energy_filtA[i] >= 400 and energy_filtA[i] <= 500: # Between the peaks
                #axtr_Abtw.plot(wavetime+incr, traces_nocosmic[tracesind_filtA[i]], label=' trace')
                #incr += 60
            #if energy_filtA[i] >= 500 and energy_filtA[i] <= 700 and i > 900: # Peak 2
                #axtr_Abtw.plot(wavetime+250, traces_nocosmic[tracesind_filtA[i]], label=' trace')
        axtr_Abtw.set_title('Traces In-Between (Green Box)')
        axtr_Abtw.set_ylabel('Voltage')
        yStart = 9000
        yEnd = 14000
        axtr_Abtw.set_xlabel('Time (ns)')
        #axtr_Abtw.set_xlim([xStart, xEnd * 2])
        axtr_Abtw.set_xlim([60, 600])
        axtr_Abtw.set_ylim([10000, yEnd])
        plt.show(block=True)  # Don't block terminal by default.

        ftr_B, axtr_B = plt.subplots(layout = 'constrained')
        # Looks for traces that fall within the gate and plot them
        for i in range(10):
            if i < len(energy_filtB): axtr_B.plot(wavetime, traces_nocosmic[tracesind_filtB[i]], label=' trace')
        axtr_B.set_title('Traces in cut B')
        axtr_B.set_ylabel('Voltage')
        axtr_B.set_xlabel('Time (ns)')
        axtr_B.set_xlim([xStart, xEnd * 2])
        axtr_B.set_ylim([yStart, yEnd])
        plt.show(block=True)  # Don't block terminal by default.

        ftr_C, axtr_C = plt.subplots(layout = 'constrained')
        # Looks for traces that fall within the gate and plot them
        for i in range(10):
            axtr_C.plot(wavetime + i*50, traces_nocosmic[tracesind_filtC[i]], label=' trace')
        axtr_C.set_title('Beta Traces')
        axtr_C.set_ylabel('Voltage')
        axtr_C.set_xlabel('Time (ns)')
        axtr_C.set_xlim([xStart, xEnd * 2])
        axtr_C.set_ylim([10000, yEnd])

        ftr_CHE, axtr_CHE = plt.subplots(layout = 'constrained')
        incr = 0
        # Looks for high energy beta traces
        for i in range(100):
            if energy_filtC[i] >= 400 and energy_filtC[i] <= 500:
                axtr_CHE.plot(wavetime + incr, traces_nocosmic[tracesind_filtC[i]], label=' trace')
                incr += 60
        axtr_CHE.set_title('High Energy Beta Traces')
        axtr_CHE.set_ylabel('Voltage')
        axtr_CHE.set_xlabel('Time (ns)')
        axtr_CHE.set_xlim([60, 600])
        axtr_CHE.set_ylim([10000, yEnd])

        # Draws the long and short gates if you want
        #LG_xline = np.linspace(86,196,2)
        #LG_yline = np.linspace(12000,12000,2)
        #newLG_xline = np.linspace(86,396,2)
        #newLG_yline = np.linspace(11500,11500,2)
        #HF_xline = [396, 396]
        #HF_yline = [9000, 14000]
        #TF_xline = [96, 1096]
        #TF_yline = [11000, 11000]
        #axtr_C.plot(LG_xline, LG_yline, color='black', linewidth = 2)
        #axtr_C.plot(newLG_xline, newLG_yline, color='blue', linewidth = 2)
        #axtr_C.plot(TF_xline, TF_yline, color='green', linewidth = 2)
        #axtr_C.plot(HF_xline, HF_yline, color='red', linestyle='dashed', linewidth = 2)

        plt.show(block=True)  # Don't block terminal by default.

    # Plots the PSD parameter vs the energy with the cut
    # Draws the straight A cuts
    xline = np.linspace(0,4096,10)
    yline = np.linspace(0,PSDlowA,10)
    #PSDlowlineA = np.full(len(xline), PSDlowA)
    PSDhighlineA = np.full(len(xline), PSDhighA)

    xline = np.linspace(0, (PSDlowA - 0.375) / (-0.0001442), 9)
    yline = np.full(len(xline), -0.0001442 * xline + 0.375)
    xline = np.append(xline, 4096)
    PSDlowlineA = np.append(yline, PSDlowA)

    #Draws the B cut
    # For regular Ultima Gold
    xlineB = np.linspace(0, (PSDhighA - 1) / (-0.00021973),9)
    PSDlineB = np.full(len(xlineB), -0.00021973 * xlineB + 1)
    xlineB = np.append(xlineB, 4096)
    PSDlineB = np.append(PSDlineB, PSDhighA)

    # For Ultima Gold AB+F with a coarse gain of 2.5 fC/(lsb x Vpp)
    if UG == 'F':
        xlineB = np.linspace(1000, 4096,10)
        PSDlineB = np.full(len(xlineB), -0.000184168 * xlineB + 1.18417)

    # For Ultima Gold AB+F with a coarse gain of 10 fC/(lsb x Vpp)
    #xlineB = np.linspace(0, (PSDhighA - 0.02 - 1) / (-0.0006),9)
    #PSDlineB = np.full(len(xlineB), -0.0006 * xlineB + 1)
    #xlineB = np.append(xlineB, 4096)
    #PSDlineB = np.append(PSDlineB, PSDhighA - 0.02)

    ElineC = np.full(len(yline), 150)

    fig_psdE2, ax_psdE2 = plt.subplots(layout = 'constrained')
    h2 = ax_psdE2.hist2d(energy_nocosmic, psd_parameter_nocosmic, bins=[nbins,500], range=[[0,4095], [0,1]], norm=mpl.colors.Normalize(), cmin = 1)
    fig_psdE2.colorbar(h2[3],ax=ax_psdE2)
    ax_psdE2.plot(xline, PSDlowlineA, color='black', linewidth = 3)
    ax_psdE2.plot(xline, PSDhighlineA, color='black', linewidth = 3)
    ax_psdE2.plot(xlineB, PSDlineB, color='red', linewidth = 3, zorder = 10)
    #ax_psdE2.plot(ElineC, yline, color='blue', linewidth = 3 )
    plt.ylim([0, 1])
    plt.xlim([0,4095])

    ax_psdE2.set_ylabel('PSD parameter')
    ax_psdE2.set_xlabel('ADC Channel')
    plt.show()

    # Focus on the B cut region
    psd_parameter_B = []
    for i in range(len(tracesind_filtB)):
        psd_parameter_B.append(psd_parameter_nocosmic[tracesind_filtB[i]])
    fig_psdEB, ax_psdEB = plt.subplots(layout = 'constrained')
    h2 = ax_psdEB.hist2d(energy_nocosmic, psd_parameter_nocosmic, bins=[nbins,500], range=[[0,4095], [0,1]], norm=mpl.colors.Normalize(), cmin = 1)
    fig_psdEB.colorbar(h2[3],ax=ax_psdEB)
    plt.ylim([0.4, 1.2])
    plt.xlim([2000,5200])
    ax_psdEB.set_ylabel('PSD parameter')
    ax_psdEB.set_xlabel('ADC Channel')
    plt.show()


    # Plots the filtered energy data for cut A
    fig_filtEA, ax_filtEA = plt.subplots(layout = 'constrained')
    (data_entries, bins, patches) = ax_filtEA.hist(energy_filtA, bins=nbins, range = [0,4095])
    ax_filtEA.set_xlabel('ADC Channel')
    ax_filtEA.set_ylabel('Counts')

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
    (data_entriesC, binsC, patchesC) = ax_filtEC.hist(energy_filtC, bins=nbins, range = [0,4095])
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
    # Fitting the alpha peaks

    # Get bin centers
    bincenters = np.array([0.5 * (bins[i] + bins[i + 1])  for i in range(len(bins) - 1)])

    # Find the bin width
    binwidth = bins[1] - bins[0]

    # Define range for each peak
    # For regular Ultima Gold
    lower_bound1 = 500
    upper_bound1 = 1050

    lower_bound2 = 1200
    upper_bound2 = 1800

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
    p01 = ([2000, 800, 200])
    p02 = ([2000, 1500, 100])

    # For Ultima Gold AB
    if UG == 'AB':
        p01 = ([2000, 900, 200])
        p02 = ([2000, 1700, 100])

    # For Ultima Gold AB + F with a coarse gain of 2.5 fC/(lsb x Vpp)
    if UG == 'F':
        p01 = ([2000, 1300, 200])
        p02 = ([2000, 2400, 200])
        lower_bound = 700
        upper_bound = 2800

        # Using optical grease
        lower_bound = 1300
        upper_bound = 4000


    # For Ultima Gold AB+F with a coarse gain of 10 fC/(lsb x Vpp)
    # lower_bound1 = 200
    # upper_bound1 = 450

    # lower_bound2 = 450
    # upper_bound2 = 750

    # Get points for each bin center
    xpeak = bincenters[bincenters > lower_bound]
    x = xpeak[xpeak < upper_bound]
    x = x.ravel()

    y = data_entries[bincenters > lower_bound]
    y = y[xpeak < upper_bound]
    y = y.ravel()

    # Set parameter guesses
    # For regular Ultima Gold
    p01 = ([2000, 800, 200])
    p02 = ([2000, 1500, 100])

    # For Ultima Gold AB
    if UG == 'AB':
        p01 = ([2000, 900, 200])
        p02 = ([2000, 1700, 100])

    # For Ultima Gold AB + F with a coarse gain of 2.5 fC/(lsb x Vpp)
    if UG == 'F':
        p01 = ([2000, 1300, 200])
        p02 = ([2000, 2400, 200])
        p0trip = ([2000, 1200, 200, 2000, 1500, 200, 2000, 1750, 200, 20])

        # Using optical grease
        #p0trip = ([2000, 1800, 200, 2000, 2500, 200, 2000, 3500, 200, 20])
        p0trip = ([2000, 1800, 200, 2000, 2200, 200, 2000, 2900, 200, 20])


    # For Ultima Gold AB+F with a coarse gain of 10 fC/(lsb x Vpp)
    # p01 = ([2000, 325, 200])
    # p02 = ([2000, 600, 100])

    # Fits the data to 3 Gaussians and plots
    E_param, Ecov = curve_fit(TripleGaussFit, xdata=x, ydata=y, p0=p0trip, maxfev=10000)
    E_err = np.sqrt(np.diag(Ecov))

    xspace = np.linspace(lower_bound, upper_bound, 1000)

    # Plots the hitogram and fitted function
    fitplt, fitax = plt.subplots(layout='constrained')
    fitax.hist(energy_filtA, bins=nbins, range=[0, 4095], label='PSD Filtered Energy')
    fitax.plot(xspace, TripleGaussFit(xspace, *E_param), linewidth=2.5, label=r'3 Gaussian Fit')
    fitax.plot(xspace, GaussFit(xspace, E_param[0], E_param[1], E_param[2]), linewidth=2.5)
    fitax.plot(xspace, GaussFit(xspace, E_param[3], E_param[4], E_param[5]), linewidth=2.5)
    fitax.plot(xspace, GaussFit(xspace, E_param[6], E_param[7], E_param[8]), linewidth=2.5)
    plt.legend()

    plt.xlim(0, 4095)
    fitax.set_xlabel('ADC Channel')
    fitax.set_ylabel('Counts')
    plt.show()

    # Calculates the integral
    GInt1, GIntErr1 = quad(GaussFit, lower_bound, upper_bound, args=(E_param[0], E_param[1], E_param[2]))
    GInt2, GIntErr2 = quad(GaussFit, lower_bound, upper_bound, args=(E_param[3], E_param[4], E_param[5]))
    GInt3, GIntErr3 = quad(GaussFit, lower_bound, upper_bound, args=(E_param[6], E_param[7], E_param[8]))

    # Correct integrals for bin width
    GInt1 = GInt1 / binwidth
    GInt2 = GInt2 / binwidth
    GInt3 = GInt3 / binwidth

    # Use counting statistics for the uncertainty: ~ sqrt(num counts)
    GIntErr1 = np.sqrt(GInt1)
    GIntErr2 = np.sqrt(GInt2)
    GIntErr3 = np.sqrt(GInt3)

    # Prints out the fitting parameters and integrals
    print("The triplet fit parameters are:")
    print("Amplitude 1: ", E_param[0], " +/-", E_err[0], " Counts")
    print("Mean 1: ", E_param[1], " +/-", E_err[1], " ADC Channel")
    print("Stdev 1: ", E_param[2], " +/-", E_err[2], " ADC Channel")
    print("Integral 1: ", GInt1, " +/-", GIntErr1, " Counts")
    print("Amplitude 2: ", E_param[3], " +/-", E_err[3], "Counts")
    print(" Mean 2: ", E_param[4], " +/-", E_err[4], "ADC Channel")
    print(" Stdev 2: ", E_param[5], " +/-", E_err[5], " ADC Channel")
    print("Integral 2: ", GInt2, " +/-", GIntErr2, " Counts")
    print("Amplitude 3: ", E_param[6], " +/-", E_err[6], "Counts")
    print(" Mean 3: ", E_param[7], " +/-", E_err[7], "ADC Channel")
    print(" Stdev 3: ", E_param[8], " +/-", E_err[8], " ADC Channel")
    print("Integral 3: ", GInt3, " +/-", GIntErr3, " Counts")

    print("")
    print(r"Total $^{225}$Ac decays: ", GInt1 + GInt2 + GInt3, " +/- ",
          np.sqrt(GInt1 + GInt2 + GInt3), " Counts")

    # Calculate the activity based on the run time
    # Could make it automatic, use timestamp[last] - timestamp[0] but that would be slightly off
    # Might be a good enough approx
    Acactivity = (GInt1 + GInt2 + GInt3) / runtime  # CPS
    Acactivity = Acactivity * (1 / 37000)  # microcurie
    # Use relative counting uncertainty to get activity uncertainty
    Acerror = np.sqrt(GInt1 + GInt2 + GInt3 + len(energy_filtC) + len(energy_filtB)) * Acactivity / (GInt1 + GInt2 + GInt3 + len(energy_filtC) + len(energy_filtB))
    print(r"Total $^{225}$Ac activity from 3 Gaussian fit: ", Acactivity, " +/- ", Acerror, " microcuries" )

    # Fits the data to 4 Gaussians and plots
    p0quad = ([2000, 1200, 200, 2000, 1400, 200, 2000, 1700, 200, 2000, 2200, 200, 0])

    # Using optical grease
    #p0quad = ([2000, 1700, 200, 2000, 2000, 200, 2000, 2500, 200, 2000, 3300, 200, 0])
    #p0quad = ([2000, 1500, 200, 2000, 1800, 200, 2000, 2300, 200, 2000, 3200, 200, 0])
    p0quad = ([2000, 1500, 200, 2000, 1800, 200, 2000, 2000, 200, 2000, 3000, 200, 0])

    E_param_quad, Ecov_quad = curve_fit(QuadGaussFit, xdata=x, ydata=y, p0=p0quad, bounds=((0,-np.inf,-np.inf,0,-np.inf,-np.inf,0,-np.inf,-np.inf,0,-np.inf,-np.inf,0),(np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf)), maxfev=100000)
    E_err_quad = np.sqrt(np.diag(Ecov_quad))

    print('Quad fit params: ', E_param_quad)

    fitplt_quad, fitax_quad = plt.subplots(layout='constrained')
    fitax_quad.hist(energy_filtA, bins=nbins, range=[0, 4095], label='PSD Filtered Energy')
    fitax_quad.plot(xspace, QuadGaussFit(xspace, *E_param_quad), linewidth=2.5, label=r'4 Gaussian Fit')
    fitax_quad.plot(xspace, GaussFit(xspace, E_param_quad[0], E_param_quad[1], E_param_quad[2]), linewidth=2.5)
    fitax_quad.plot(xspace, GaussFit(xspace, E_param_quad[3], E_param_quad[4], E_param_quad[5]), linewidth=2.5)
    fitax_quad.plot(xspace, GaussFit(xspace, E_param_quad[6], E_param_quad[7], E_param_quad[8]), linewidth=2.5)
    fitax_quad.plot(xspace, GaussFit(xspace, E_param_quad[9], E_param_quad[10], E_param_quad[11]), linewidth=2.5)
    #plt.legend()

    plt.xlim(0, 4095)
    fitax_quad.set_xlabel('ADC Channel')
    fitax_quad.set_ylabel('Counts')
    plt.show()

    #Using 4 Gaussian fit to calculate activity
    # Calculates the integral
    GInt1, GIntErr1 = quad(GaussFit, lower_bound, upper_bound, args=(E_param_quad[0], E_param_quad[1], E_param_quad[2]))
    GInt2, GIntErr2 = quad(GaussFit, lower_bound, upper_bound, args=(E_param_quad[3], E_param_quad[4], E_param_quad[5]))
    GInt3, GIntErr3 = quad(GaussFit, lower_bound, upper_bound, args=(E_param_quad[6], E_param_quad[7], E_param_quad[8]))
    GInt4, GIntErr4 = quad(GaussFit, lower_bound, upper_bound, args=(E_param_quad[9], E_param_quad[10], E_param_quad[11]))

    # Correct integrals for bin width
    GInt1 = GInt1 / binwidth
    GInt2 = GInt2 / binwidth
    GInt3 = GInt3 / binwidth
    GInt4 = GInt4 / binwidth

    # Use counting statistics for the uncertainty: ~ sqrt(num counts)
    GIntErr1 = np.sqrt(GInt1)
    GIntErr2 = np.sqrt(GInt2)
    GIntErr3 = np.sqrt(GInt3)
    GIntErr4 = np.sqrt(GInt4)

    # Prints out the fitting parameters and integrals
    print("The quad fit parameters are:")
    print("Amplitude 1: ", E_param_quad[0], " +/-", E_err_quad[0], " Counts")
    print("Mean 1: ", E_param_quad[1], " +/-", E_err_quad[1], " ADC Channel")
    print("Stdev 1: ", E_param_quad[2], " +/-", E_err_quad[2], " ADC Channel")
    print("Integral 1: ", GInt1, " +/-", GIntErr1, " Counts")
    print("Amplitude 2: ", E_param_quad[3], " +/-", E_err_quad[3], "Counts")
    print(" Mean 2: ", E_param_quad[4], " +/-", E_err_quad[4], "ADC Channel")
    print(" Stdev 2: ", E_param_quad[5], " +/-", E_err_quad[5], " ADC Channel")
    print("Integral 2: ", GInt2, " +/-", GIntErr2, " Counts")
    print("Amplitude 3: ", E_param_quad[6], " +/-", E_err_quad[6], "Counts")
    print(" Mean 3: ", E_param_quad[7], " +/-", E_err_quad[7], "ADC Channel")
    print(" Stdev 3: ", E_param_quad[8], " +/-", E_err_quad[8], " ADC Channel")
    print("Integral 3: ", GInt3, " +/-", GIntErr3, " Counts")
    print("Amplitude 4: ", E_param_quad[9], " +/-", E_err_quad[9], "Counts")
    print(" Mean 4: ", E_param_quad[10], " +/-", E_err_quad[10], "ADC Channel")
    print(" Stdev 4: ", E_param_quad[11], " +/-", E_err_quad[11], " ADC Channel")
    print("Integral 4: ", GInt4, " +/-", GIntErr4, " Counts")

    Acactivity = (GInt1 + GInt2 + GInt3 + GInt4) / runtime  # CPS
    Acactivity = Acactivity * (1 / 37000)  # microcurie
    # Use relative counting uncertainty to get activity uncertainty
    Acerror = np.sqrt(GInt1 + GInt2 + GInt3 + GInt4 + len(energy_filtC) + len(energy_filtB)) * Acactivity / (GInt1 + GInt2 + GInt3 + GInt4 + len(energy_filtC) + len(energy_filtB))
    print(r"Total $^{225}$Ac activity from 4 Gaussian fit: ", Acactivity, " +/- ", Acerror, " microcuries" )
    print(r"Total alpha counts: ", GInt1 + GInt2 + GInt3 + GInt4 + len(energy_filtB))


# Runs the main file
main()