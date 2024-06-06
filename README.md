# Alpha-Beta-Pulse-Shape-Discrimination-for-Nuclear-Medicine
### Johnathan Phillips - jphillipsps1@gmail.com
  
## Purpose
Detector unpacking and analysis code for the detection of decays from candidites for alpha therapy using pulse-shape discrimination.

## Data
The data files read into this code are not publicly available. Please reach out to me via email if you would like to procure them.

## Requirements
The python packages used in this file are as follows:
* numpy
* matplotlib
* scipy
* pathlib
  
This repository only contains the python file. All packages and python environments must be handled by the user.

## Acknowledgements 

### Advisors:
* Lee Sobotka [WashU]
* Robert Charity [WashU]
### Collaborators:
* Daniel Thorek [WashU Med]
* Abbie Hasson [WashU Med]

## User Guide

### Data Format:
This program uses the numpy genfromtxt() command to unpack the data files. The data is in the form of digitized waveforms, with 248 samples across 496 ns. Each event contains in order:
* The board number (never changes).
* The channel number (0 for a cosmic ray veto detector, 1 for the LSC).
* The event timestamp (in ps).
* The integrated energy from the long gate.
* The integrated energy from the short gate.
* The pulse-shape discrimination (PSD) parameter calculated from the long and short energies.
* The waveform samples.

Especially for large files, processing the waveforms drastically increases the run time. Two switches are provided:
* alldat: Set to True to process all events. False for a certain number (default is 10000).
* wavesdat: Set to True to process waveforms. False to skip waveform samples.

I would not advise turning both "alldat" and "wavesdat" to "True" as the unpacker will take a very long time.

### Event Types:
There are three major categories for events, depending on the scintillant used. There are then three options for running the code by changing the variable "UG" to one of three values:
* UG = 'R' : Event uses the regular Ultima Gold. Any filename that does not contain UG' ' uses the regular Ultima Gold.
* UG = 'AB' : Event uses the AB Ultima Gold.
* UG = 'F' : Event uses the F Ultima Gold. This is really a mixture of the F and AB variants as F alone is not for aqueous samples.
