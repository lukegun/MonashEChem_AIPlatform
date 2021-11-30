import time
import datetime
import os
import numpy as np
import pandas as pd
import Script_generator as Scripgen
import MECwrit_mod as MECr
import plotting_scripts as plotter
import MultiharmVsV_mod as MHplot
import matplotlib.pyplot as plt
import psycopg2

inputMEC = 'masterin/MasterE27Hz.txt'
data = MECr.mecreader(inputMEC)

def NN_ILgenerator_dc(current,Nx,Ny,data):
    n_i = int(Ny )
    n_e = int(Nx)
    n_sig_fig = 4

    Estart = float(data.iloc[2])
    Erev = float(data.iloc[3])
    Ncycles = float(data.iloc[4])
    dpoints = len(current)

    for i in range(int(Ncycles)):
        DC1 = np.linspace(Estart,Erev,int(dpoints/(2*Ncycles)))
        DC2 = np.linspace(Erev, Estart, int(dpoints / (2 * Ncycles)))
        if i == 0:
            DC = np.append(DC1,DC2)
        else:
            x = np.append(DC1, DC2)
            DC = np.append(DC,x)

    e_dc = DC

    min_i = min(current)
    max_i = max(current)
    del_i = (max_i - min_i) / (n_i - 1)
    min_e = e_dc.min()
    max_e = e_dc.max()
    del_e = (max_e - min_e) / (n_e - 1)
    i_bin = [int((current[i] - min_i) / del_i) for i in range(0, len(current))]
    e_bin = [int((e_dc[i] - min_e) / del_e) for i in range(0, len(e_dc))]
    # build structure
    ml_output = np.zeros((n_i, n_e))
    # increment for each bin member
    for i in range(0, len(e_dc)):
        ml_output[i_bin[i], e_bin[i]] += 1
    # reshape to single array
    ml_output = ml_output.reshape(n_i * n_e)
    # scale
    max_count = ml_output.max()
    # / max, * 10^sig fig
    ml_output = [(10 ** n_sig_fig) * ml_output[i] / max_count for i in range(0, len(ml_output))]
    # to int via ceiling
    ml_output = np.ceil(ml_output)
    # / 10^sig fig
    ml_output = [round(ml_output[i] / (10 ** n_sig_fig), n_sig_fig) for i in range(0, len(ml_output))]
    ml_output = np.array(ml_output)
    # save to text file
    ml_reshaped = ml_output.reshape((n_i, n_e))

    return ml_reshaped

def NN_ILgenerator_harmflip(current,Nx,Ny):
    #tim3 = time.time()
    n_i = int(Ny * 0.5)
    n_e = int(Nx * 2)
    n_sig_fig = 4

    e_dc = np.linspace(0,100,len(current))

    min_i = min( current)
    max_i = max(current)
    del_i = (max_i - min_i) / (n_i - 1)
    min_e = e_dc.min()
    max_e = e_dc.max()
    del_e = (max_e - min_e) / (n_e - 1)
    i_bin = [int((current[i] - min_i) / del_i) for i in range(0, len(current))]
    e_bin = [int((e_dc[i] - min_e) / del_e) for i in range(0, len(e_dc))]

    # build structure
    ml_output = np.zeros((n_i, n_e))
    # increment for each bin member
    for i in range(0, len(e_dc)):
        ml_output[i_bin[i], e_bin[i]] += 1
    # reshape to single array
    ml_output = ml_output.reshape(n_i * n_e)
    # scale
    max_count = ml_output.max()
    # / max, * 10^sig fig
    ml_output = [(10 ** n_sig_fig) * ml_output[i] / max_count for i in range(0, len(ml_output))]
    # to int via ceiling
    ml_output = np.ceil(ml_output)
    # / 10^sig fig
    ml_output = [round(ml_output[i] / (10 ** n_sig_fig), n_sig_fig) for i in range(0, len(ml_output))]
    ml_output = np.array(ml_output)

    # save to text file
    ml_reshaped = ml_output.reshape((n_i, n_e))

    forward = ml_reshaped[:,0:int(n_e/2)]
    reverse = ml_reshaped[:, int(n_e/2):]
    print(forward.shape)
    print(reverse.shape)
    ml_reshaped = np.concatenate((np.flip(forward),reverse),axis=0)

    return ml_reshaped #np.flip(ml_reshaped, 0) #np.transpose(ml_reshaped)


