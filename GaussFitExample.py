
########################################################
# Code by Johnathan Phillips
# 2/26/2025
#
# NOTE: This is an example cutout of the analysis code.
#       The listed code only handles fitting alpha peaks.
#       This code will not work standalone, it must be
#       integrated into a larger analysis code.
#
########################################################


# Define the Gaussian Fit Function.
# This is a separate function outside of the Main() function.
def GaussFit(x, a, mu, s):

    return a * np.exp(-((x - mu)**2)/(2 * s**2))

# Two fit more than one peak, simply add Gaussians together
def DoubleGaussFit(x, a, mu, s, a1, mu1, s1):

    return GaussFit(x,a,mu,s) + GaussFit(x,a1,mu1,s1)


# The actual fitting occurs inside the Main() function and after
#   the data has been unpacked and processed.
nbins = 512 # Define some number of bins.

# First, create the histogram for the alpha spectra using the numpy array "energy".
fig, ax = plt.subplots(layout = 'constrained')
# data_entries holds the "y" information, bins will provide the "x" information.
# The (x,y) data is needed to turn a histogram into a plot that can be fit.
# This is extracted from the alpha histogram.
(data_entries, bins, patches) = ax.hist(energy, bins=nbins, range = [0,4095])

# Define lower and upper fitting bounds based on your alpha energy spectrum.
# If you are fitting multiple Gaussians added you only need one lower and one
#   upper bound.
# For Pb-212 the peaks are very separated, so I will fit them with separate
#   Gaussians, requiring
#   two sets of bounds.
lower_bound1 = 500 # First peak
upper_bound1 = 1100
lower_bound2 = 1000 # Second peak
upper_bound2 = 1800

# Get points for each bin center within the lower and upper fitting bounds.
# The bin center gives the "x" and the counts in that bin give the "y", giving
#   me (x,y) points.
xpeak1 = bincenters[bincenters>lower_bound1] # First peak
x1 = xpeak1[xpeak1<upper_bound1]
x1 = x1.ravel()
y1 = data_entries[bincenters>lower_bound1]
y1 = y1[xpeak1<upper_bound1]
y1 = y1.ravel()

xpeak2 = bincenters[bincenters>lower_bound2] # Second peak
x2 = xpeak2[xpeak2<upper_bound2]
x2 = x2.ravel()
y2 = data_entries[bincenters>lower_bound2]
y2 = y2[xpeak2<upper_bound2]
y2 = y2.ravel()

# Make initial guesses for parameters.
# The amplitude is not too important and the peak position and stdev should be
#   constant for a sample.
p01 = ([2000, 800, 200]) # First peak
p02 = ([2000, 1500, 100]) # Second peak

# Fit the peaks to Gaussians using the scipy curve_fit functions.
# Without parameter bounds, the default fitting method is least squares using the
#   Levenberg-Marquardt algorithm
E_param1, E_cov1 = curve_fit(GaussFit, xdata=x1, ydata=y1, p0=p01, maxfev=10000) # First peak
E_param2, E_cov2 = curve_fit(GaussFit, xdata=x2, ydata=y2, p0=p02, maxfev=10000) # Second peak
E_err1 = np.sqrt(np.diag(E_cov1)) # Gives statistical error for the first fit
E_err2 = np.sqrt(np.diag(E_cov2)) # Give the above for the second fit

# If the fitting parameters need to be bounded, I would fit like this.
# Here I bound the amplitude and position to be greater than zero, which is always true
# With parameter bounds, the default fitting method is least squares using the
#   Trust Region Reflective algorithm
# E_param, E_cov = curve_fit(GaussFit, xdata=x, ydata=y, p0=p0,
#   bounds=((0,0,-np.inf),(np.inf,np.inf,np.inf)), maxfev=10000)

# Create "x" points for each fit that I use to get "y" points and plot.
xspace1 = np.linspace(lower_bound1, upper_bound1, 1000)
xspace2 = np.linspace(lower_bound2, upper_bound2, 1000)

# Plot fitted functions
ax.plot(xspace1, GaussFit(xspace1, *E_param1), linewidth=2.5, label=r'Peak 1 fit')
ax.plot(xspace2, GaussFit(xspace2, *E_param2), linewidth=2.5, label=r'Peak 2 fit')

# Change how the plot looks
plt.legend(loc='upper right', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(0, 4095)
fitax.set_xlabel('ADC Channel', fontsize=20)
fitax.set_ylabel('Counts', fontsize=20)
plt.show()

# To extract the fitting parameters and get position and standard deviation, I can print
#   (or write to file) E_param.
print("Mean 1: ", E_param1[1], " ADC Channel")
print("Stdev 1: ", E_param1[2], " ADC Channel")
print(" Mean 2: ", E_param2[1], "ADC Channel")
print(" Stdev 2: ", E_param2[2], " ADC Channel")