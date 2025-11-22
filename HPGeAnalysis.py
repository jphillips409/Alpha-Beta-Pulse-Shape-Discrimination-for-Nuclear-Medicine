########################################################
# Created on December 21 2024
# @author: Johnathan Phillips
# @email: j.s.phillips@wustl.edu

# Purpose: To unpack and analyze Pb-212, Ra-223, and Ac-225 HPGe data

# Detector: Coaxial high-purity germanium gamma ray detector
########################################################

# Import necessary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
import matplotlib as mpl
from pathlib import Path

def main():

    filepath = Path(r"/Users/jphillips409/Documents/ThorekPb212")

    PbFile = r'Pb_HPGeData.csv'
    RaFile = r'Ra_HPGeData.csv'
    AcFile = r'Ac_HPGeData.csv'

    PbFile_toOpen = filepath / PbFile
    RaFile_toOpen = filepath / RaFile
    AcFile_toOpen = filepath / AcFile

    PbData = np.genfromtxt(PbFile_toOpen, skip_header=True, delimiter=',')
    RaData = np.genfromtxt(RaFile_toOpen, skip_header=True, delimiter=',')
    AcData = np.genfromtxt(AcFile_toOpen, skip_header=True, delimiter=',')

    pltPb, axPb = plt.subplots(layout = 'constrained')
    axPb.plot(PbData[:,0],PbData[:,1], label='Linear Fit', color='blue',linewidth = 1)
    axPb.set_xlabel('Energy (keV)')
    axPb.set_ylabel('Counts')
    plt.xlim(0,3400)
    plt.ylim(1,10000)
    axPb.set_yscale('log')

    plt.savefig(r"C:\Users\jphillips409\Documents\ThorekPb212\Pb212_HPGe.eps", format='eps')
    plt.savefig(r"C:\Users\jphillips409\Documents\ThorekPb212\Pb212_HPGe.png", format='png')
    pltPb.show()

    pltRa, axRa = plt.subplots(layout = 'constrained')
    axRa.plot(RaData[:,0],RaData[:,1], label='Linear Fit', color='Green', linewidth = 1)
    axRa.set_xlabel('Energy (keV)')
    axRa.set_ylabel('Counts')
    plt.xlim(0,1300)
    plt.ylim(1,30000)
    axRa.set_yscale('log')

    plt.savefig(r"C:\Users\jphillips409\Documents\ThorekPb212\Ra223_HPGe.eps", format='eps')
    plt.savefig(r"C:\Users\jphillips409\Documents\ThorekPb212\Ra223_HPGe.png", format='png')
    pltRa.show()

    pltAc, axAc = plt.subplots(layout = 'constrained')
    axAc.plot(AcData[:,0],AcData[:,1], label='Linear Fit', color='Red', linewidth = 1)
    axAc.set_xlabel('Energy (keV)')
    axAc.set_ylabel('Counts')
    plt.xlim(0,2100)
    plt.ylim(1,30000)
    axAc.set_yscale('log')

    plt.savefig(r"C:\Users\jphillips409\Documents\ThorekPb212\Ac225_HPGe.eps", format='eps')
    plt.savefig(r"C:\Users\jphillips409\Documents\ThorekPb212\Ac225_HPGe.png", format='png')
    pltAc.show()

main()