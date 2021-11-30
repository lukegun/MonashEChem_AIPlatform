import numpy as np
import pandas as pd
import sys
import time
# pickle for training/testing data storage
import pickle
import matplotlib.pyplot as plt

# just some function for generating the
def NN_ILgenerator(current):
    #tim3 = time.time()
    n_i = 100
    n_e = 100
    n_sig_fig = 4

    e_dc = np.linspace(0,100,len(current))

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
    #format_txt = '%.' + str(n_sig_fig) + 'f'
    #np.savetxt("harmtestpost.csv", ml_reshaped[::-1,:], delimiter=",", fmt=format_txt)

    binarydata = ml_reshaped[::-1,:].tobytes()      # This weird correction is here so that time series is in the right order
    #print(time.time() - tim3)

    return binarydata

# connect to server and get an list of all reaction IDs and paired with class of form react_class = [[ReactionID,Reactionmech]]
def image_gen_4N(cuurr,imagedimensions,modeltype,n_channels):

    minI = -max(abs(cuurr))
    maxI = max(abs(cuurr))

    DC_datasweep1 = NN_ILgenerator_dc_exp(cuurr[:int(len(cuurr)/4)], imagedimensions[0], imagedimensions[1],minI,maxI)


    DC_datasweep1 = DC_datasweep1.reshape(1, imagedimensions[1], imagedimensions[0], 1)

    """
                        Required to get into a format that is easy for Keras (and Tensorflow underneath) to use. Format is:

                        (n_images, IMAGE_X, IMAGE_Y, n_channels)

                        where n_images is the number of images to stack on top of each other in a single batch or the entire training 
                        set, the image has resolution X,Y and the number of channels is 1 for black and white or 3 for RGB colour images.            
    """

    if modeltype != "gareth":

        DC_datasweep = DC_datasweep1

        for i in range(n_channels- 1):

            DC_datasweep2 = NN_ILgenerator_dc_exp(cuurr[(i+1)*int(len(cuurr) / n_channels):(i+2)*int(len(cuurr) / n_channels)], imagedimensions[0],
                                                imagedimensions[1],minI,maxI)

            DC_datasweep2 = DC_datasweep2.reshape(1, imagedimensions[1], imagedimensions[0], 1)
            DC_datasweep = np.concatenate((DC_datasweep,DC_datasweep2),axis = 3)

    else:
        DC_datasweep = DC_datasweep1

    return DC_datasweep

def NN_ILgenerator_dc_exp(current,Nx,Ny,minI,maxI):
    n_i = int(Ny)
    n_e = int(Nx)
    n_sig_fig = 4

    Estart = 0.67
    Erev = -0.67
    #print("WARNING Estaet: " + str(Estart) +" and " + "Erev: " + str(Erev))
    Ncycles = 1
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

    min_i = minI
    max_i = maxI
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
