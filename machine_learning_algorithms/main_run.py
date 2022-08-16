#placeholderpythonfile
import matplotlib.pyplot as plt
import pandas as pd

#Removes all printed stuff
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import DNNtimecluster_mod as TC_mod
import TrainDNNtimeclass_mod as TDNN
import DC_neuralnetwork_mod as NNt
import ImagebasedDNN as IB_NNt
import time
import sys
import numpy as np
import DNN_run_mods
import DNN_run_mods as runner_mod
import window_func as wint
from joblib import load
from scipy.fftpack import rfft
import tensorflow as tf
from tensorflow import keras


# python3 main_run.py AG4AC1.txt AG4AC8.txt AG4AC10.txt -dnn reactmech rocket -p D=6.5E-6cm2/s A=3.36e-02cm2 C=1.0mM k0=100cm/s
# where the thing after -dnn [[reactmech,clustering],[rocket,inceptiontime]]

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])

#Removes all printed stuff
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


print(sys.argv)

expfile,DNNarch,parameterestimates = runner_mod.terminalextract(sys.argv)

# extracts a constant dic from terminal line
paraestdic = runner_mod.paraextract(parameterestimates)

print(expfile)
print(DNNarch)
print(parameterestimates)

#"""Something to load in some settings"""
# these are currently placeholder
serverdata, supivisorclass, ACCase, variate, DNNmodel, cpu_workers, gpu_workers, deci, harmdata, modelparamterfile, \
    trainratio, reactionmech = TDNN.inputloader("testDC.txt")
print(ACCase)


""" Get experimental parameters """
if DNNarch[2] == "AC":
    ACmode, Evalues, ACsettings, scanrate, currenthold,expvoltage, timedata= runner_mod.ACinpureader(expfile)
elif DNNarch[2] == "DC":
    ACmode,Evalues,ACsettings,scanrate,currenthold, polarity,expvoltage, timedata = runner_mod.DCinpureader(expfile)

"""LOAD IN ALL THE DNN SPECIFIC PARAMETERS CAN USE THE TRAINING FILE FOR THIS"""
maxharmonicsinDNN = 9

if DNNarch[2] == "AC":
    DNNlength = 2**8
elif DNNarch[2] == "DC":
    DNNlength = 2 ** 8

""" extract the harmonics """
""" Treat experimental data into specific format """
if ACmode:
    DNNdeci = int(len(currenthold[0]) / DNNlength)
    std = 0.1
    AutoNharm = 0.5
    HarmMax = 12
    #print("pooo")
    #print(ACsettings)
    freqband = ACsettings[0]*3/5
    #print(freqband)
    bandwidth = []
    for j in range(2):
        band = []
        for i in range(12): # this needs to be 12
            band.append(freqband)
        bandwidth.append(band)

    # generate a bandwidth approx for 8 harmonics as a function of frequency
    nexpdeci,frequens = runner_mod.Simerharmtunc(len(currenthold[0]), timedata[0], bandwidth, 12, [ACsettings[0]])
    #print(nexpdeci)
    #Count number of harmonics
    Nharm, fft_res = runner_mod.harmoniccounter(currenthold[0], nexpdeci, AutoNharm)

    #this is here to check to see if Nharm is more then the allowed harmonics
    if maxharmonicsinDNN <= Nharm:
        Nharm = maxharmonicsinDNN

    bandwidthplot = np.array(bandwidth)
    bandwidth = bandwidthplot[:,:Nharm]

    # generate filters
    #simfilterhold = wint.RGguassconv_filters(2**Nsim, bandwidth, timedata[1]/(2**Nsim), [ACsettings[0]], std)
    expfilterhold = wint.RGguassconv_filters(len(currenthold[0]), bandwidth, timedata[0], [ACsettings[0]], std)

    # generate all harmonics
    hil_storehold = []
    fundharmhold = []
    for current in currenthold:
        fft_res = rfft(current)
        hil_store,fundimentalharm = runner_mod.Harmonic_gen(fft_res, expfilterhold)
        hil_storehold.append(hil_store)
        fundharmhold.append(fundimentalharm)

    #average the fundimental for capacitance stuff
    fundimentalharm = np.average(fundharmhold, axis=0)
    del fundharmhold

    # average each harmonic
    xy = []
    for i in range(Nharm + 1):
        harms = []
        for hils in hil_storehold:
            harms.append(hils[i][:])
        harm = np.average(np.array(harms), axis=0)
        xy.append(harm)
    hilstore = xy

    percentagetrun = 0.05
    truntime = percentagetrun*timedata[1]

    trunvec_sim, trunvec_exp = runner_mod.EXPharmtreatment(len(hilstore[0]), timedata[1], [truntime,  timedata[1] - truntime ], 2**16)
    del trunvec_sim

    #stored for the capacitance fitting before truncatio
    capcurrent = hilstore[1]
    xy = []
    for hils in hil_store:
        harm = wint.Average_noiseflattern(hils, trunvec_exp, avgnum=20)
        xy.append(harm[::DNNdeci]) # adds decimation in

    #decimate the harmonics to the number used in the DNN
    current = xy[:maxharmonicsinDNN ]
    "truncate edges"

    del  hil_storehold

else: # DC mode
    # use a non float so error of trancation can be removed throughout
    DNNdeci = len(currenthold[0]) / DNNlength

    current = np.average(currenthold,axis= 0)

    # small moving average to nemove some noise
    current = runner_mod.moving_average(current,5)

    x = np.zeros(DNNlength)
    #error on truncation is spread through the time series
    for i in range(DNNlength):
        x[i] = current[int(i*DNNdeci)]

    # stored for the capacitance fitting before truncation
    current = x #current[::DNNdeci] #add the decimation in

    #this may cause errors but required to


    del currenthold # clear up ram
    fundimentalharm = None # required due to ac process
    ACsettings = None


"""Check the DC for the case where """
#""" Get experimental parameters """
if ACmode:
    Evaluehold = []
    for file in expfile:
        Evalues, ACsettingsb,scanrateb,blah,expvoltageb,timedatab = runner_mod.expextract(file)
        Evaluehold.append(Evalues)



    # check to see if exp files are the same
    if all([Evaluehold[0] == x1 for x1 in Evaluehold]):
        Evalues = Evaluehold[0]
        # """Average the experimental voltage""" something can be added here for the psuedo reference

    else:
        print("ERROR the exp settings files are wrong and don't all match up")
        exit()

# if oxidation sweep flip the DC current
if Evalues[0] == Evalues[1] and Evalues[0] == Evalues[2]:
    if ACmode:
        pass
    else:
        currenthold = [current]  # this is a lazy dumby
elif Evalues[0] == Evalues[1] and Evalues[0] == Evalues[3]: #flip DC
    if ACmode:
        current[0] = -1*current[0]
    else:
        current = -1*current
        currenthold = [current] # this is a lazy dumby
else:
    print("ERROR: the evalues aren't in cycle format and hasn't been set up yet")
    print(Evalues)
    exit()


#flip DC to Match the training data

""" NON_dimensionalise the current """ #need to consider the Non dim and effect of surface and diff difference and the effect on this
print("Below normalisation doesn't work right remove")

scurr, constant = runner_mod.currentnondim(currenthold[0],paraestdic.get(6), paraestdic.get(22), paraestdic.get(21))
temp = 273.15 + 23
surfaceconfinedcurr, Sconstant = runner_mod.currentnondimsurfconfined(currenthold[0],paraestdic.get(6), paraestdic.get(21), scanrate/1000, temp)

if ACmode:
    Gencurrent = runner_mod.normalisationrange(current,constant)
    surfcurrent = runner_mod.normalisationrange(current,Sconstant)

    # adds the empty array in so no divide by zero issue
    if Nharm < maxharmonicsinDNN:
        for i in range(len(hilstore), maxharmonicsinDNN):  # this adds the blank harmonics if none where found
            Gencurrent.append(np.zeros(DNNlength))
            surfcurrent.append(np.zeros(DNNlength))
    testvolt = False
else: # DC normalisations
    # the lists here is to format the system into the DNN
    # this is the DC specific normalisation stuff
    xs = [-1, 1]
    current = (xs[1] - xs[0]) * ((current + np.max(np.abs(current))) / (2*np.max(np.abs(current)))) + xs[0]
    #current = (xs[1] - xs[0]) * ((current - np.min(current)) / (np.max(current) - np.min(current))) +xs[0]
    #current = current / max(current)
    # x = (xs[1] - xs[0]) * ((x - np.min(x)) / (np.max(x) - np.min(x)) + xs[0])
    testvolt = False
    if testvolt:
        nd = int(DNNlength/2)
        current = np.array([current[:nd], current[-1:nd-1:-1]])

    Gencurrent = current #[constant*current]
    surfcurrent = current #[Sconstant*current]


# will need a correction for when this isn't the case


"""Load in a settings file with the location of the DNN models"""
if ACmode: # AC mode
    DNNfile,clusterbayesloc = DNN_run_mods.DNN_model_locloader("DNN_MODELS/AC_DNNMODEL_LOCATION.txt")
else: # DC case
    DNNfile,clusterbayesloc = DNN_run_mods.DNN_model_locloader("DNN_MODELS/DC_DNNMODEL_LOCATION.txt")


"""LOAD IN THE CLUSTERING FILE"""
if type(clusterbayesloc) != None:
    clusterbayes = DNN_run_mods.clusterbayesloader(clusterbayesloc)
    print(clusterbayes)
    bayesmatrix = np.zeros((len(clusterbayes[0]),len(clusterbayes)))
    for i in range(len(clusterbayes)):
        j = 0
        for key,items in clusterbayes[i].items():
            bayesmatrix[j,i] = items
            j += 1

"""SET UP A COPY OF THE DATA FOR ALL AVALIBLE MODES"""
modeltype = list(DNNfile.keys())
print(modeltype)
if any("R"==i[0] for i in modeltype): # rocket classifiers are used
    dic = {}
    for i in range(len(Gencurrent)):
        dic.update({str(i): Gencurrent[i]})

    print(len(Gencurrent))
    Rscurr = pd.DataFrame([dic])
    print(Rscurr.shape)
if any("I"==i[0] for i in modeltype): # inception time classifiers are used
    print(type(Gencurrent))
    Iscurr =np.array([Gencurrent])
    print(Iscurr.shape)
    r = Iscurr.shape
    if testvolt:
        Iscurr = Iscurr.transpose(0, 2, 1)
    else:
        if ACmode:
            Iscurr = Iscurr.transpose(0, 2, 1)
            #Iscurr = Iscurr.reshape((1, r[1], r[0]))
        else:
            Iscurr = Iscurr.reshape((1,r[1],r[0]))

for i in range(r[0]):
    if testvolt:
        plt.figure()
        plt.plot(Iscurr[0,:,i])
        plt.plot(Iscurr[0, :, 1])
        plt.savefig(str(i) + ".png")
        plt.close()
    else:
        indnnplot = True
        if ACmode:
            print(Iscurr.shape)
            for ii in range(9):
                plt.figure()
                plt.plot(Iscurr[0, :, ii])
                plt.savefig(str(ii) + ".png")
                plt.close()

            #optional plot for ploting
            if indnnplot:
                from matplotlib.colors import ListedColormap, LinearSegmentedColormap
                from seaborn import heatmap
                import seaborn as sn
                #plt.figure()
                p = heatmap(np.transpose(np.array(Iscurr)[0,:,:])) # this is peak lazy
                p.set_xlabel("datapoint ($t_i$)")
                p.set_ylabel("Harmonic #")
                #plt.imshow()
                #p.colorbar()
                plt.savefig("cmap_AC.png")

                plt.close()
        else:
            plt.figure()
            plt.plot(Iscurr[0, :, i])
            plt.savefig(str(i)+".png")
            plt.close()

"""LOAD THE DNN MODEL """
from sktime.transformations.panel.rocket import Rocket
print(DNNfile)
if ACmode:
    DNNfile = {'IC': 'DNN_MODELS/AC/inceptiontime_clustering.hdf5','IR': 'DNN_MODELS/AC/inceptiontime_reactmech.hdf5'}#
else:
    DNNfile = {'IC': 'DNN_MODELS/DC/inceptiontime_clustering.hdf5','IR': 'DNN_MODELS/DC/inceptiontime_reactmech.hdf5'}  #,'IR': 'DNN_MODELS/DC/inceptiontime_reactmech.hdf5'

predictkey ={}
for key, items in DNNfile.items():

    if key[0] == "I":
        model = keras.models.load_model(items)

        #model.summary()

        # Run inference on CPU
        with tf.device('/cpu:0'):
            y_pred = model.predict(Iscurr)
        # load the inception time model

        # set input data to Rocket mode

    elif key[0] == "R":
        model = load(items, None) #might need a correction here for .joblib file
        # load the Rocket model
        kernelloc = items.split(".")
        kernelloc = kernelloc[0]+ "_kernel."+kernelloc[1]
        print(kernelloc)
        rocket = load(kernelloc,None) #  fix

        """NEED TO LOAD IN ROCKET AND CLASSIFIER"""
        transformedcurr = rocket.transform(Rscurr)
        y_pred = model.predict(transformedcurr)
        print("shit")
        print(y_pred)
        # set input data to Rocket mode
        # do prediction
        """For testing remove later"""


    else:
        print("ERROR MODEL TYPE IS NOT SUPPORTED")
        exit()

    # Normalise the prediction for the sigmoid function
    #print("Using sigmoid correction L311")
    #y_pred[0] = y_pred[0] / sum(y_pred[0])

    #print prediction
    if key[1] == "R": # reaction mech labels
        y_pred = y_pred[0] # extract the one output


    elif key[1] == "C": #clastering label reaction mechanisms
        r,c = bayesmatrix.shape
        contintable = np.zeros((r,c))
        bayetot = []
        """THIS IS GOING TO BE A NIGHTMARE"""
        if key[0] == "R":
            #print(bayesmatrix)
            bayetothold = []
            #print(y_pred[0], bayesmatrix[:, y_pred[0]])
            bayetothold.append(bayesmatrix[:, y_pred[0]])
            y_pred = bayetothold
            maxpred = []
            #print(bayetothold)
        elif key[0] == "I":

            print("assuming sigmoid fitting function")
            #print(y_pred)
            #y_pred[0] = y_pred[0]/sum(y_pred[0])

            #print(bayesmatrix)
            for i in range(r):
                for j in range(c):
                    contintable[i,j]= y_pred[0][j]*bayesmatrix[i,j]
                bayetot.append(sum(contintable[i,:]))
        #print(contintable)
        print(y_pred)
        clusterprob0 = y_pred
        y_pred = bayetot

    # This doesn't seem to work in any real way'
    if True and key[0] == "I" and ACmode:
        DNN_run_mods.incept_imagegen(model,Iscurr,ACmode,key,file.split("/")[-1])


    print(expfile)
    print(key, y_pred)
    predictkey.update({key:y_pred})

if 'IC' in DNNfile:
    clusterprob = clusterprob0[0]
    reactprob = bayetot

    """DEFINE BELOW AS FUNCTION"""
    # select two most probale cluster groups

    nxm = list(clusterprob).index(max(clusterprob))
    nym = list(clusterprob).index(runner_mod.second_largest(clusterprob))

    if clusterprob[nym] > clusterprob[nxm]/2:
        clusterind = [nxm, nym]
    else:
        clusterind = [nxm]

    # SELECT two MOST Probable reaction mechanism
    nxm = list(reactprob).index(max(reactprob))
    nym = list(reactprob).index(runner_mod.second_largest(reactprob))


    if reactprob[nym] > reactprob[nxm]/2:
        reactind = [nxm, nym]
    else:
        reactind = [nxm]

    # This is the label for looking
    reactclust = []
    for x in reactind:
        for y in clusterind:
            reactclust.append([x,y])

            # load the react mechs
    if ACmode:
        f = open("DNN_MODELS/AC/clusteringlabelsbayes.txt","r")
    else:
        f = open("DNN_MODELS/AC/clusteringlabelsbayes.txt", "r")

    readlines = f.readlines(1)
    readlines = readlines[0].split("\t")
    reactlabels = []
    for stuff in readlines[1:-1]:
        x = stuff.split(":")
        reactlabels.append(x[0])
    f.close()

    # print this to like
    RUNSQL = False
    if ACmode and RUNSQL:
        # load the cluster labels
        f = open("DNN_MODELS/AC/clusteringnumberlabelinfo.txt", "r")
        readlines = f.readlines()
        clusterlabels = []
        for stuff in readlines:
            x = stuff.split("\t")
            clusterlabels.append([x[2].split(" ")[1]]+ x[3:-1])
        f.close()

        """GET THE PRIOR DATA IF INCEPTION TIME """
        print(reactclust)
        # load the reaction ID where cluster label and reaction mechanism
        IDstot = []
        parametertot = []
        for stuff in reactclust:

            #function to get the IDS from database
            IDreact = DNN_run_mods.sqlReactIDclusters(serverdata,clusterlabels[stuff[1]],reactlabels[stuff[0]])
            # load the parameter distrabutions of where the react IDs are
            parametervalues = DNN_run_mods.sqlparameterclusters(serverdata, IDreact)
            IDstot.append(IDreact)
            parametertot.append(parametervalues)

    """# sort out the parameter distrabutiuons based of the reaction mechanisms"""

    # will also need to fix this all up for use on runner in a GCP setting

    """ PRINT THE OUTPUT for a person"""

f = open("output_dnnrunner.txt","a+")
f.write(expfile[0]+":\n")
for key, reactprob in predictkey.items():
    f.write(key+": ")
    print(key, end=",\t")
    for i in range(len(reactlabels)):
        print(reactlabels[i]+":"+str(int(reactprob[i]*100)),end="%,\t")
        f.write(reactlabels[i]+":"+str(int(reactprob[i]*100))+"%,\t")
    f.write("-\t")
    if key == "IC":
        for x in reactclust:
            f.write(str(reactionmech[x[0]])+",\t"+str(x[1])+",\t")
    else:
        f.write(str(reactionmech[np.argmax(reactprob)])+",\t")
    f.write("\n")
    print("")
f.close()

"""Print something to pass to the greedy output"""
