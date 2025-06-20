########################################################
# Created on March 7 2025
# @author: Johnathan Phillips
# @email: j.s.phillips@wustl.edu

# Purpose: To unpack and analyze data for the Pb-212 decay from an HPGe detector

########################################################

# Import necessary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
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

def Unpack(name, cal, runtime):

    #For getting the actual data
    if cal is False and runtime is False: data = np.genfromtxt(name, skip_header = 12, skip_footer = 15, usecols = (0))

    #For getting the calibration
    if cal is True and runtime is False: data = np.genfromtxt(name, skip_header = 16404, skip_footer = 6, usecols = (0,1))

    if cal is False and runtime is True: data = np.genfromtxt(name, skip_header = 16401, skip_footer = 9, usecols = (0))

    return data

def Gauss(x,a,mu,s):
    return a * np.exp(-((x - mu)**2)/(2 * s**2))

def GaussplusBack(x,a,mu,s,b):
    return Gauss(x,a,mu,s) + b

def main():

    # Read in the HPGe data in the form of .spe files
    filepath = Path(r"C:\Users\j.s.phillips\Documents\Thorek_PSDCollab\Pb-212-HPGe-LOD-selected")

    data_file = r'BLANK.spe'
    #data_file = r'day0-600s-43502pmtime-22625date.spe'
    #data_file = r'day1-600s-111423amtime-22725date.spe'
    #data_file = r'day1-600s-050140pmtime-22725date.spe'
    #data_file = r'day2-1800s-104055amtime-22825date.spe'
    #data_file = r'day2-1800s-052540pmtime-22825date.spe'
    #data_file = r'day3-3600s-110748amtime-30125date.spe'
    #data_file = r'day4-10800s-014657pmtime-30225date.spe'
    #data_file = r'day5-21600s-100016amtime-30325date.spe'
    #data_file = r'day6-43200s-111227amtime-30425date.spe'
    #data_file = r'day7-43200s-095533amtime-30525date.spe'
    #data_file = r'day8-43200s-111316amtime-30625date.spe'



    file_to_open = filepath / data_file

    #Extract data
    data = Unpack(file_to_open, False, False)

    #Get runtime and cal [1] is x0 [0] is x1
    calval = Unpack(file_to_open, True, False)
    runtime = Unpack(file_to_open, False, True)

    print(calval)
    print(runtime)

    #Make an array of x-points to match the lengh of data
    xspace = np.linspace(0,len(data)-1,len(data))
    xcal = xspace*calval[1] + calval[0]

    #Plot the raw spectrum
    #rawplt, rawax = plt.subplots(layout='constrained')
    #rawax.plot(xspace,data)

    #plt.ylim(0)
    #plt.xlim(0)
    #rawax.set_ylabel('Counts')
    #rawax.set_xlabel('ADC Channels')

    #plt.show()

    #Plot the calibrated spectrum - zoom in if desired
    calplt, calax = plt.subplots(layout='constrained')
    calax.plot(xcal,data)

    plt.xlim(560,600)
    plt.ylim(0,40)
    calax.set_ylabel('Counts')
    calax.set_xlabel('Energy (keV)')

    #Fit the gaussian
    # Define range for Gauss fit
    lower_bound = 580
    upper_bound = 590

    # Get points for each bin center
    x = xcal[xcal < upper_bound]
    x1 = x[x > lower_bound]
    x1 = x1.ravel()

    y = data[xcal < upper_bound]
    y1 = y[x > lower_bound]
    y1 = y1.ravel()

    # Set parameter guesses
    p0 = ([20, 583, 0.06,0]) # With flat background
    #p0 = ([20, 2612, 0.06]) # No Background

    # Fits the data to double Gaussian and plots
    param, cov = curve_fit(GaussplusBack, xdata = x1, ydata = y1, p0 = p0, bounds=((0,lower_bound,-np.inf,0),(np.inf,upper_bound,np.inf,np.inf)), maxfev = 10000)
    #param, cov = curve_fit(Gauss, xdata = x1, ydata = y1, p0 = p0, bounds=((0,lower_bound,-np.inf),(np.inf,upper_bound,np.inf)), maxfev = 10000)
    err = np.sqrt(np.diag(cov))

    xspaceFit = np.linspace(lower_bound, upper_bound, 1000)

    #calax.plot(xspaceFit,Gauss(xspaceFit,param[0],param[1],param[2]))
    calax.plot(xspaceFit,GaussplusBack(xspaceFit,param[0],param[1],param[2],param[3]))

    plt.show()


    #Do a simple sum of the counts in the spectrum
    ctotal = 0
    for i in range(len(data)):
        ctotal += data[i]


    #Do a sum for peak while subtracting a constant bkg
    ptotal = 0
    for i in range(len(y1)):
        ptotal += y1[i] - param[3]

    #Just get gaussian portion of integral
    GInt1, GIntErr1 = quad(Gauss,lower_bound, upper_bound, args=(param[0],param[1],param[2]))

    GInt1 = GInt1/(xcal[1]-xcal[0])

    print("HPGe Data:")
    print("Peak energy (keV): ",param[1])
    print("Total number of counts in spectrum: ",ctotal)
    print("Run time (s) ",runtime)
    print("Gaussian integral: ", GInt1)
    if len(param) == 4: print("Flat background: ", param[3])

    print("Sum of peak bounds: ", sum(y1))
    print("Sum of peak minus bkg: ", ptotal)



main()