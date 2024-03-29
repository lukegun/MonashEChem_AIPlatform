# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 23:12:19 2021

@author: luke
"""

import numpy as np
from scipy import signal
import scipy.stats as stats
import scipy as sci
import matplotlib.pyplot as plt
from scipy.signal import hilbert as anal_hil_tran # NEED signal to make a smooth envolope
from scipy.fftpack import rfft, irfft, rfftfreq

def square_window_fund(ftt_freq,band):
    
    #funcdimental is full input
    nhigh = find_nearest(ftt_freq, band)
    
    x = np.zeros(len(ftt_freq))
    
    x[0:nhigh] = 1
    
    return x

def square_window(ftt_freq,mu,band):
    
    nlow = find_nearest(ftt_freq, mu-band/2)
    nhigh = find_nearest(ftt_freq, mu+band/2)
    
    x = np.zeros(len(ftt_freq))
    
    x[nlow:nhigh] = 1
    
    return x

def unit_guassian(x,mu,std):
    return np.exp(-0.5*((x[:]-mu)/std)**2)

def find_nearest(array1, value):
    array1 = np.asarray(array1)
    idx = (np.abs(array1 - value)).argmin()
    return int(idx)

# below works but isnt relavant at moment
def blackman_win(x, band, mu):
        
    a_0 = 7938/18608
    a_1 = 9240/18608
    a_2 = 1430/18608
    
    print(mu-band)
    
    z = a_0 - a_1*np.cos((2*np.pi*(x[:]- mu- band)/(2*band))) + a_2*np.cos((4*np.pi*(x[:]- mu- band)/(2*band)))
    
    int_s = find_nearest(x, mu-band)
    int_e = find_nearest(x, mu+band)
    
    print(int_e)
    
    x =  np.zeros(len(x))
    
    x[int_s:int_e] = z[int_s:int_e]
    
    return x

def analitical_RGguassconv(x,band,mean,std):
    

    s =  np.sqrt(2)/2 # np.sqrt(2)/2
    c = np.sqrt(np.pi/2) # np.sqrt(np.pi/2)
    
    z = c*std*(sci.special.erf((-x[:]+mean+band/2)*s/std) - sci.special.erf((-x[:]+mean-band/2)*s/std))
    
    # normalization for ease of computation
    z = z/max(z)
    
    return z

def analitical_RGguassconv_fund(x,band,std):
    

    s =  np.sqrt(2)/2 # np.sqrt(2)/2
    c = np.sqrt(np.pi/2) # np.sqrt(np.pi/2)
    
    z = c*std*(sci.special.erf((-x[:]+band)*s/std) - sci.special.erf((-x[:]-band)*s/std))
    
    # normalization for ease of computation
    z = z/max(z)
    
    return z

""" Doesn't work
def analitical_blackman_win(x, band, mu):
    
    a_0 = 7938/18608
    a_1 = 9240/18608
    a_2 = 1430/18608
    
    print(mu-band)
    xneg = -x[:]+ mu
    xpos = -x[:]+ mu + 2*band
    
    K = (2*np.pi)/2*band
    
    z =2*a_0*band -  a_1/K*(np.sin(K*xpos[:]) - np.sin(K*xneg[:])) + a_2/K*(np.sin(2*K*xpos[:]) - np.sin(2*K*xneg[:]))
    
    int_s = find_nearest(x, mu-2*band)
    int_e = find_nearest(x, mu+2*band)
    
    print(int_e)
    
    x =  np.zeros(len(x))
    
    x[int_s:int_e] = z[int_s:int_e]
    
    return x"""

def windowed_harm_gen(fft_res, bandwidth, Nac, filters):

    hil_store = Harmonic_gen(fft_res, filters, Nac, bandwidth)  # extract envolope data

    return hil_store
    
    
# generates harmonics envolopes from MECSim current output    
def Harmonic_gen(fft_res, filters, Nac, bandwidth): # cuts the harmonic fourier space data out {NEEDS TO EXCLUDE FUNDEMENTAL}

    nharm = len(filters)     # counts the number of harmonics
    #Np = len(fft_res)    # gets the number of datapoints
    # harmstore = np.zeros(nharm*spaces[4] ,int(Mwin*(2/g)))
    hil_store = []
    #N_ifft = np.zeros((nharm*Nac + 1))

    #hil_store[0,:] = irfft(y)  # generates fundimental harmonics from cut window
       

    x = fft_res*filters[0] #This need to be truncated in the future
            
    harmonic = irfft(x)   # generates harmonics
    
    hil_store.append(harmonic)
    i = 1
    while i!= nharm:

        x = fft_res*filters[i] #This need to be truncated in the future
            
        harmonic = irfft(x)   # generates harmonics
            
        hil_store.append(abs(anal_hil_tran(harmonic))) #uses HILBERT TRANSFORM to generate the envolope

        # using the abs fixed an issue with the complexes disapearing in the return
        i += 1
 
    return hil_store

# flaterns the noise of DC and fundimental then replaces it of averge cut for capacitance
def Average_noiseflattern(harmonic,trunvec,avgnum):
    
    int_s = trunvec[0]	# first truncation point
    int_e = trunvec[1]	# second truncation point
    
    # adjustment for DC
    harmonic[:int_s] = np.average(harmonic[int_s+1:int_s+avgnum])
    harmonic[int_e:] = np.average(harmonic[int_e-avgnum:int_e])
    
    return harmonic

#
def noiseflattern(hil_store,time,trun):
    
    Ndc = 2 # here to exclude DC and fundimenal
    
    # finds indexes
    int_s = trunvec[0]  # first truncation point
    int_e = trunvec[1]  # second truncation point
    
    # truncates to zero
    hil_store[Ndc:,:int_s] = 0
    hil_store[Ndc:,int_e:] = 0
    
    return hil_store
