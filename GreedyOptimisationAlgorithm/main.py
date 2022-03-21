import shutil

import numpy as np
import matplotlib.pyplot as plt
import pandas
import pandas as pd
import time
import os
import sys
import greedy_mod as gm
import GreedyAlgo as GrAl
import numpy as np
import window_func as wint
import CapacitanceFit as cap
import scriptformmod as sm
import CMAES as standCMA
import plotting_scripts as plotter
import ML_signal_processing as MLsp
import priortracker as prtr
from timeseries_modules import Harm_sigma_percentge
from scipy.fftpack import rfft
from inputwrittermod import inputwritter
from copy import deepcopy

# extracts varibles from the command line
expfile,reactmech,parameterestimates = gm.terminalextract(sys.argv)

print("NEED TO FIX UP RESISTANCE")

#checks to see if reaction mech labels is based of supivied labels (Which are supported) or unsupivised
if any(reactmech[0] == i for i in ["E","EE","EC"]):
    supervisedlabels = True # this is for the prior allocation which are stored differently for the two methods

# extracts a constant dic from terminal line
paraestdic = gm.paraextract(parameterestimates)

""" load default settings file """
numiter, stdaccept = gm.settingsload("greedy_algo_settings.txt")

""" Get experimental parameters """
Evaluehold = []
ACsettingshold = []
scanratehold = []
currenthold = []
voltagehold = []
for file in expfile:
    Evalues, ACsettings,scanrate,current,expvoltage,timedata = gm.expextract(file)
    Evaluehold.append(Evalues)
    ACsettingshold.append(ACsettings)
    scanratehold.append(scanrate)
    currenthold.append(current)
    voltagehold.append(expvoltage)


# check to see if exp files are the same
if all([Evaluehold[0] == x1 for x1 in Evaluehold]) and all([ACsettingshold[0] == x1 for x1 in ACsettingshold]) and all([scanratehold[0] == x1 for x1 in scanratehold]):
    Evalues = Evaluehold[0]
    ACsettings = ACsettingshold[0]
    scanrate = scanratehold[0]
    # """Average the experimental voltage""" something can be added here for the psuedo reference
    expvoltage = np.average(voltagehold,axis= 0)

else:
    print("ERROR the exp settings files are wrong and don't all match up")
    exit()

""" Identify if AC OR DC """
ACmode = False
if ACsettings[1] != 0:
    ACmode = True

#Nsimlength, exptime = gm.timeextract()
# allocate a simulated number of data based of some if functions
Nsim = gm.simaccuracy(timedata[1], ACmode,ACsettings[0])

""" Treat experimental data into specific format """
if ACmode:

    std = 0.1
    AutoNharm = 0.3
    HarmMax = 12
    #print("pooo")
    #print(ACsettings)
    freqband = ACsettings[0]*3/5
    #print(freqband)
    bandwidth = []
    for j in range(2):
        band = []
        for i in range(12):
            band.append(freqband)
        bandwidth.append(band)

    # generate a bandwidth approx for 8 harmonics as a function of frequency
    nexpdeci,frequens = gm.Simerharmtunc(len(currenthold[0]), timedata[0], bandwidth, HarmMax, [ACsettings[0]])
    #print(nexpdeci)
    #Count number of harmonics
    Nharm, fft_res = gm.harmoniccounter(currenthold[0], nexpdeci, AutoNharm)

    bandwidthplot = np.array(bandwidth)
    bandwidthplot = bandwidthplot[:,:Nharm]


    # generate filters
    simfilterhold = wint.RGguassconv_filters(2**Nsim, bandwidth, timedata[1]/(2**Nsim), [ACsettings[0]], std)
    expfilterhold = wint.RGguassconv_filters(len(currenthold[0]), bandwidth, timedata[0], [ACsettings[0]], std)

    # generate all harmonics
    hil_storehold = []
    fundharmhold = []
    for current in currenthold:
        fft_res = rfft(current)
        hil_store,fundimentalharm = gm.Harmonic_gen(fft_res, expfilterhold)
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

        harm = np.average(np.array(harms),axis= 0)
        xy.append(harm)
    hilstore = xy

    percentagetrun = 0.05
    truntime = percentagetrun*timedata[1]

    trunvec_sim, trunvec_exp = gm.EXPharmtreatment(len(hilstore[0]), timedata[1], [truntime,  timedata[1] - truntime ], 2**Nsim)

    #stored for the capacitance fitting before truncation
    capcurrent = hilstore[1]

    xy = []
    for hils in hilstore:
        harm = wint.Average_noiseflattern(hils, trunvec_exp, avgnum=20)
        xy.append(harm)
    current = xy

    del  hil_storehold

else: # DC mode
    current = np.average(currenthold,axis= 0)

    # stored for the capacitance fitting before truncation
    capcurrent = current
    del currenthold # clear up ram
    fundimentalharm = None # required due to ac process
    ACsettings = None
# majority of the above can be saved to the training DNN then loaded from
""" Fit double layer capacitance automatically """
capfitclass = cap.overallcapfit(capcurrent,ACmode,trunvec_exp,timedata[0],Evalues,scanrate,expvoltage,fundimentalharm,ACsettings)
capparas, Icdlfit, E0values = capfitclass.overall_fitting() # gets the cap parameter values, background charging current and E0values


# correction from farads to farads /cm**2
invArea = 1/paraestdic.get(6)
for i in range(len(capparas) - 1):
    capparas[i] = capparas[i]*invArea

# collects all possible parameters to be fit for the reaction mechanism
allpara,scaledparas,functparas,Mspaces,Nreact = gm.Reactparaallocator(reactmech[0])

# generates an output file
split = expfile[0].split("/")
x = split[-1].split(".")[0]
filenametot = gm.outputfilegenertor("Greedyoutput_"+reactmech[0]+"_"+x)


""" collects good estimates of the pinned parameters and priors """ # I'll need a setting files of the grouped parameters
# if this takes a long time we'd have to do this seperatly for the program
serverdata = ["postgres", "Quantum2", "35.193.220.74", 5432,"ftacv-test"] # This all needs to be taken from a settings file
if supervisedlabels == True: # this gets prior information from the sql database
    priors = gm.supivisedpriors(serverdata, reactmech[0],allpara)
else: # case of unsupivised labels
    print("worry about this later")

# get a starting MECSim file
#some function to define what to put in varible locations

pinnedvalues, var,priorsdel = gm.varallocator(allpara,paraestdic,priors,scaledparas,functparas,E0values,Mspaces)

var = pd.DataFrame(var)

data = sm.mastergenericloader(reactmech[0])

data = sm.values2Master(data,Evalues, ACmode,ACsettings, scanrate,Nsim,pinnedvalues,Mspaces,capparas)

# set up the top CMAES {} file and formating

# ignore DC as an option
ignoreDC = True
if ignoreDC == True:
    x = 0
else:
    x = 1
harm_weights = [x] #DC
for i in range(Nharm):
    harm_weights.append(1)

#cuts the bandwidth and others down to size
for i in range(len(bandwidth)):
    bandwidth[i] = bandwidth[i][:Nharm] # cuts nandwidth down to a useable length
    #truncates the filterhold to identified amount
    expfilterhold = expfilterhold[:Nharm+1]
    simfilterhold = simfilterhold[:Nharm+1]
    current = current[:Nharm+1]

cap_series = [True]
x = []
for i in range(len(capparas) - 1):
    x.append(capparas[i])
cap_series.append(x)

op_settings = [0, 0, 0, 0, 0]  # prealocation
op_settings[0] = int(1)  # This one setts the optimization method
print("op settings for Ncore needs to be a varible setting")
op_settings[1] = int(4)  # this one does the num cores
datatype = int(1)  # Gets the type of data were compairing against assume FTACV
print("this needs to be a automatic setting")
op_settings[2] = 2 ** float(14)  # sets the number of datapoints to compair in the current
op_settings[3] = float(0.025)  # sets the tolx value might increase to 0.1 if too slow
op_settings[4] = float(0.33)  # sets the initial sigma

# changes to scalar func and func to format
x =[[len(scaledparas)]]
for i in scaledparas:
    x.append(np.array(i))
scaledparas = x

#make the last input in funct and scaled change to the data row FIX
if len(functparas) != 0: # this is for some standardisation
    x =[functparas[0]]
    for i in functparas[1:]: #skip the initial value
        x.append(np.array(i)) #as types needed as the numpy stuff will automatically set it to int if else
    functparas = x
else:
    functparas = [[0]]

# sets up AC settings in the old format
AC_freq = np.array([ACsettings[0]]) # just set up for one atm
AC_amp = np.array([ACsettings[0]]) # just set up for one atm

DCAC_method = gm.ACDC_method(AC_amp, op_settings)

# issolates the functional varible
funcvar_holder = gm.functionholder(var)

# sort the below out
scaledparas = gm.scaledline_assigner(Mspaces, scaledparas)

#adjusts the experimental current so this only needs to be done once
current= np.array(current)
plotcurrent = current # here for plotting stuff
trunstime = [truntime,timedata[-1]-truntime]
deci = int(len(current[0,:])/op_settings[2])
trunvec_sim = [int(trunvec_sim[0]/int(2**Nsim/op_settings[2])),int(trunvec_sim[1]/int(2**Nsim/op_settings[2]))]
current = current[:,::int(deci)]
current = current[:,trunvec_sim[0]:trunvec_sim[1]]

# checks to see if theres multiple files if there is tries bayes optimisation
if len(expfile) == 1:
    sigma = None  # uses this as well use HarmPer # can always use Bayes_expharmper
    header = ["STANDARD", "HarmPerFit", "CMAES"]
    bayelogic = False
else:
    Harmonicwindowing = ["Conv_guassian", std] # can also be ["squarewindow",0]
    header = ["STANDARD", "Bayes_ExpHarmPerFit", "CMAES"]
    # this is dodge but avoids me rewritting old old code
    EXPperrErr, sigma, EXPperrmean = Harm_sigma_percentge(pandas.DataFrame(expfile), np.array(current), [truntime,timedata[-1]-truntime], [ACsettings[0]], bandwidth,
                                                                   Mspaces, op_settings, Harmonicwindowing)
    bayelogic = True

# this is here to remove pinned varibles and correct all the other varible lines done in varallocator
for value in priorsdel:
    mean = var.iloc[:][0].values # THIS WORKS BUT ITS GENERALLY TO AVOID REDOING STUFF
    scaledparas,functparas,funcvar_holder,var,data,cutdic = GrAl.listmodifier(value, scaledparas, functparas, funcvar_holder, var,data,mean,Mspaces)

# Results are the best fit and logtot are all trails tried
#Results are in format [-1,1]

"""WE USE CMA ES due to previous use, a multimodalular bee algorithm would probably work better but would be more data hungry and require more CPUs"""

timetot = time.time()
itteration = 0
stdhold = []
meanhold = []
varsave = []
complete = False

# deep copy just in case
scaledparasOG = deepcopy(scaledparas)
functparasOG = deepcopy(functparas)
funcvar_holderOG = deepcopy(funcvar_holder)

#prior changer/ policy definer
priortrackerclass = prtr.optpriortracker(reactmech,var)
cutdichold = {}

while not complete:

    freq_set = {'harm_weights': harm_weights, 'bandwidth': bandwidth}  # here to allow functional ACDC functionality

    MEC_set = {'data': data, 'var': var, 'spaces': Mspaces, 'DCAC_method': DCAC_method, 'Exp_data': current,
               'scalvar': scaledparas, 'funcvar': functparas, 'funcvar_holder': funcvar_holder,
               'cap_series': cap_series, 'op_settings': op_settings, 'Nsimdeci': trunvec_sim,
               'harm_weights': harm_weights, 'bandwidth': bandwidth, 'Exsigma': sigma, 'AC_freq': AC_freq,
               'filters': simfilterhold}

    #adds the parameters used in bayes logic
    if bayelogic:
        MEC_set.update({"EXPperrErr":EXPperrErr})

    print(MEC_set)

    # Fix this to make it most optimal
    Nd = var.shape[0] # number of dimensions
    Nfunc = int(4 + 3 * np.log(Nd)) # this is number of functions per CMA
    if itteration == 0:
        Niter = 3
        rawarray,array, results,meanerror = GrAl.convergencetest(var, op_settings, header, MEC_set, current,Niter)

    else:
        Niter = 2
        rawarray,array,results,meanerror = GrAl.convergencetest(var, op_settings, header, MEC_set, current,Niter)

        # add the fitted version to reduce loop time
        meanerror.append(greederror)
        for ii in range(var.shape[0]):
            array[ii].append(greedfit[ii])
            rawarray[ii].append(rawgreedfit[ii])


    rawstd = []
    std = []
    mean = []
    i = 0
    for values in array:
        print("parametervarence")
        m = np.mean(values)
        mean.append(m)  # can check if something has converged
        RSD = np.std(values) / abs(m)
        rawstd.append(np.std(rawarray[i]))
        print(rawarray[i])
        print(values)
        if RSD >= 1:  # case where it flys of to infinite
            std.append(1)
        else:
            std.append(RSD)
        i += 1

    varsave.append(var)
    stdhold.append(std)
    meanhold.append(mean)

    varibleshold = var
    print("fuck")
    """std = [0.6713074125367431, 8.594800131993373e-05, 0.039528782294206824, 1, 0.07921235535320782, 0.0399734900941651, 1, 0.08878977338017986]
    rawstd = [0.3707133233575879, 0.00019340071339179172, 0.4793330218463139, 0.8562518871823168, 0.7864373575874074, 0.4643752846564397, 0.8942711771222993, 0.9138477560419466]
    mean = [32.94179979230941, 4.500412119457176e-06, 0.3637853184639315, 2.686110630959187, 0.4964107897566512, 0.34851243929102743, 3.3100705021635393, 0.5146131819308937]
    rawarray = [[-.99,-.99,-.99], [-0.9999987126219845, -0.9998485794674564, -0.9995345287247954], [0.9998967839491028, -0.09524645755967893, 0.0856964454451832], [-0.9999844355905539, 0.9362221779370964, -0.7301509811500242], [0.9998727747299162, -0.3499519396511086, -0.8652734496797345], [0.2388063369550644, -0.8266872559379145, 0.05093977352705413], [0.9970580723625445, 0.9999554004551342, 0.9999554004551342], [0.9537413772336654, -0.9999906922511742, 0.9230402308711299]]
    array = [[47.23205777028966, 49.88795187986882, 1.7053897267697586], [4.500002574756031e-06, 4.500302841065086e-06, 4.500930942550408e-06], [0.38387875426405854, 0.35102445701879503, 0.3564527441089409], [0.010000537590107213, 8.0229348751987, 0.025396480088753642], [0.5499936387364959, 0.48250240301744457, 0.4567363275160133], [0.36104604085423736, 0.32908123306744796, 0.355410043951397], [9.898903910599834, 0.010001540532357248, 0.021306055358427108], [0.5476870688616833, 0.4500004653874413, 0.5461520115435565]]
    """
    print(std)
    print(rawstd)
    print(mean)
    print("array stuff")
    print(rawarray)
    print(array)

    var, noshift,twobpinned = priortrackerclass.priorchange(var,rawstd,std,rawarray)
    print("testing stuff")
    print(noshift)
    print(twobpinned)

    # plotting stuff is here for the diagnostics
    Scurr = MLsp.Iterative_MECSim_Curr(*[mean], **MEC_set)

    # gets a linear time array
    Exptimearray = np.linspace(0, timedata[-1], len(plotcurrent[0, :]))

    Simtimearray = np.linspace(0, timedata[-1], len(Scurr))

    plotter.harmplot(filenametot + '/harmonicplots_I' + str(itteration), Scurr, Simtimearray, plotcurrent,
                     Exptimearray, bandwidthplot, AC_freq, Mspaces)

    # save a bunch of iteration values to folder
    GrAl.itterationprinter(filenametot + '/harmonicplots_I' + str(itteration) + "/itterationvalues.txt", itteration,
                           var, mean, std, rawstd, array,noshift)


    #pinn a value to mean and remove from
    for i in range(len(twobpinned)):
        if twobpinned[i]:
            std[i] = 5 # this is lazy but puts the shift in dominant position
        elif all([i != j  for j in noshift]):
            std[i] =  2  # this is lazy but puts the shift in non dominant position
            # pin to mean
            # remove from var

    #Make an exception here so if std is avoe this amount skip everthing and pin those parameters
    #If standard deviation hasn't been meet  build a bunch of varibles of the trees

    # Scaning over a uniform distrabution between -1 and 1
    # A completely uninformative distrabution would have std sqrt(4/12) aprox =  0.6
    # A half uninformative distrabution would have std sqrt(1/12) aprox = 0.3
    # as such 0.4 was set to be mostly uninformative enough parameter is mostly random
    maxstdvalue = 0.8 # make a loaded in parameter (parameter is varied between -1 and 1)
    maxRSD = 4 #

    logsensitive = [i for i in var.iloc[:][3].values]  # etracts is the log sensitivity of parameters

    # below fails
    for i in range(len(std)):
        if logsensitive[i] == 0: # not log sensitive system
            std[i] = 0
        else: # log sensitive system
            rawstd[i] = 0

    #boolrawstd, boolRSD
    boolrawstd = [i > maxstdvalue for i in rawstd]
    boolRSD = [i > maxRSD for i in std]
    Nstd = sum(boolrawstd) + sum(boolRSD)


    # in case therre isnn't many parameters left it just pins the worst
    if var.shape[0] == 5 and (Nstd != 1 or Nstd != 0):
        Nstd = 2
    elif var.shape[0] <= 4 and Nstd != 0:
        Nstd = 1

    """NEED TO ADD SOMETHING HERE TO JUST DO A BASIC OPT AND SET PARAMETERS IF priors have been shifted DO FIRST ONE"""

    if Nstd == 0:
        if any(jj > 0.5 for jj in rawstd) or any(jj > 0.5*3 for jj in std):
            print(itteration)
            print("rawstd")
            print(rawstd)
            print("std")
            print(std)
            rawgreedfit,greedfit,greederror,var,scaledparas,functparas,funcvar_holder,data,cutdic = GrAl.greedypin(var, scaledparas, functparas,
                                                                                    funcvar_holder,op_settings, header,
                                                                                    MEC_set, current,rawstd,
                                                                                    std,boolrawstd, boolRSD,mean)

            cutdichold.update(cutdic)

            MEC_set.update({'data': data})
            MEC_set.update({'var': var})
            MEC_set.update({'scalvar': scaledparas})
            MEC_set.update({'funcvar': functparas})
            MEC_set.update({'funcvar_holder': funcvar_holder})
            """need another thing for pinning to approx mean"""
        else: # greedy algorithm has succseded
            complete = True

    elif Nstd == 1 or Nstd == 2: #if its only bad for some just pin those
        stdind,data = GrAl.stdallocator(rawstd,std, Nstd,boolrawstd, boolRSD,mean,var,Mspaces,MEC_set.get("data"))
        for values in stdind:
            scaledparas, functparas, funcvar_holder, var,data,cutdic = GrAl.listmodifier(values,scaledparas,functparas,funcvar_holder,var,data,mean,Mspaces)
            cutdichold.update(cutdic)

        # Need to do a single optimisation here to pass to above
        # update the MEC_set dictionary
        MEC_set.update({'data': data})
        MEC_set.update({'var': var})
        MEC_set.update({'scalvar': scaledparas})
        MEC_set.update({'funcvar': functparas})
        MEC_set.update({'funcvar_holder': funcvar_holder})
        #print(scaledparas, functparas, funcvar_holder, var)
        # optimisation loop
        results, logtot = standCMA.STAND_CMAES_TOTCURR(var, op_settings, header[1], MEC_set, current)
        rawgreedfit = results[0]

        greedfit, greederror = standCMA.CMA_output([rawgreedfit], **MEC_set)


    else: # if std is bad skip below step and just pin worst 3
        stdind,data = GrAl.stdallocator(rawstd,std, 3,boolrawstd, boolRSD,mean,var,Mspaces,MEC_set.get("data"))


        for values in stdind:
            scaledparas, functparas, funcvar_holder, var,data,cutdic = GrAl.listmodifier(values, scaledparas, functparas,
                                                                               funcvar_holder, var,data,mean,Mspaces)
            cutdichold.update(cutdic)

        print(scaledparas, functparas, funcvar_holder, var)
        # Need to do a single optimisation here to pass to above
        # update the MEC_set dictionary
        MEC_set.update({'data': data})
        MEC_set.update({'var': var})
        MEC_set.update({'scalvar': scaledparas})
        MEC_set.update({'funcvar': functparas})
        MEC_set.update({'funcvar_holder': funcvar_holder})

        # optimisation loop
        results, logtot = standCMA.STAND_CMAES_TOTCURR(var, op_settings, header[1], MEC_set, current)

        rawgreedfit = results[0]

        greedfit, greederror = standCMA.CMA_output([rawgreedfit], **MEC_set)

        # Need to do a single optimisation here to pass to above

    # gets the dimensionalised values and error

    # exception if we run out of parameters to pin and test
    if var.shape[0] == 1 or var.shape[0] == 0: # zero is real bad but may occur
        complete = True

    itteration += 1 # counts the number of iteration layers

    # Might want something here to plot the convergence of the parameters

"""ADD SOMETHING HERE TO DO A FINAL HIGH PRECISSION OPTIMISATION USING THE PINNED PARAMETERS"""

# converts to accutual parametrs
var_out, Perr = standCMA.CMA_output(results, **MEC_set)

"""CMA-ES output add below stuff to it"""

print("Greedy Algo Time: " + str((time.time()-timetot)/60))
print(allpara)
print(var)
print(mean)
print(itteration)
print("results")
print(varsave)
print(stdhold)
print(meanhold)
print(var_out, Perr)


Scurr = MLsp.Iterative_MECSim_Curr(*[var_out], **MEC_set)

# gets a linear time array
Exptimearray = np.linspace(0,timedata[-1],len(plotcurrent[0,:]))
Simtimearray = np.linspace(0,timedata[-1],len(Scurr))

decivolt = int(len(Exptimearray)/len(Simtimearray))

plotter.harmplot(filenametot+'/harmonicplots', Scurr, Simtimearray, plotcurrent, Exptimearray, bandwidthplot, AC_freq, Mspaces)
#completion_time = (time.time() - t1) / 60

# print a bunch of outputs to file
""" Print to a final output for person """

# add the mean tvalues to data and print out the Masterfile
with MLsp.cd(filenametot): # this is here as old me couldnt code
    MLsp.MECSiminpwriter(mean, **MEC_set)

#NEED TO CONSIDER THE DIFFERENT LOGIC METHODS

"""FIX THE REPEAT LINE IN VAR scalar and so on"""
scaledparas = MEC_set.get('scalvar')
funcvar = MEC_set.get('funcvar')
var,scaledparas,funcvar = GrAl.varconverter(var,scaledparas,funcvar,Mspaces)

#update mecset with these values
MEC_set.update({"var":var,"funcvar":funcvar,"scalvar":scaledparas})

# prints the mean current trace to a mecoutput file which is the same
startpart = GrAl.MECoutsetter(data,AC_freq, AC_amp)

GrAl.outputwriter(filenametot+'/MECsim_output.txt',startpart,expvoltage[::decivolt],Scurr,Simtimearray)

#copy the data file to a new
if True: # this needs to a loaded setting
    for file in expfile:
        x = file.split("/")
        shutil.copy(file, filenametot+'/'+x[-1])

""" Print to a final output for pipline """
""" Set up CMA ES input """
#input loader modified CMAES
if bayelogic:
    logicm = "Bayes_ExpHarmPerFit"
else:
    logicm = "HarmPerFit"

winfunc = [1,0.1] #convulutional guass with std of 0.1
settingsdic = {"Ncore": op_settings[1],"Ndata":int(np.log2(op_settings[2])) ,"tolx":0.01 , "sigma":op_settings[4] }

inputwritter(filenametot+'/CMAES_output.txt',"CMAES",logicm,data,expfile,var,winfunc,settingsdic,MEC_set,trunstime)

""" Set up BI input file """
#input loader modified BI

if bayelogic:

    print("fix priors")
    print("add starting point as mean")
    varshift = GrAl.bayespriorshift(var,mean)
    print(varshift)
    """NEED SOMETHING TO PICK PROPER PRIOR RANGES"""
    settingsdic = {"Ncore": 4,"Ndata":int(np.log2(op_settings[2])) ,"Nprior": 250 , "chaintot":25000,"burnin":0.25 }
    inputwritter(filenametot+'/ADMCMC_output.txt',"ADMCMC",logicm,data,expfile,varshift,winfunc,settingsdic,MEC_set,trunstime)

    #sets up one from all paras, ie ideally should fit it
    print(cutdichold)
    cutvarhold = pd.DataFrame(cutdichold).transpose()

    Np = len(cutvarhold)
    rempara = []
    remparamean = []
    meandel = []
    for i in range(Np):
        code = cutvarhold.iloc[i][4]
        row = cutvarhold.iloc[i][5]

        if int(code / 10) == 2:
            x = row - Mspaces[1] + 1
        elif int(code / 10) == 3:
            x = row - Mspaces[2] + 1
        elif int(code / 10) == 4:
            x = row - Mspaces[2] + 1
        else:
            x = row

        cutvarhold.loc[cutvarhold.index[i],5] = x

    print("test")
    print(scaledparasOG)
    Np = len(scaledparasOG)
    rempara = []
    remparamean = []
    meandel = []
    for i in range(1,Np):
        code = scaledparasOG[i][2]
        row = scaledparasOG[i][3]

        if int(code / 10) == 2:
            x = row - Mspaces[1] + 1
        elif int(code / 10) == 3:
            x = row - Mspaces[2] + 1
        elif int(code / 10) == 4:
            x = row - Mspaces[2] + 1
        else:
            x = row
        print(i,x)
        scaledparasOG[i][3] = x


    print(scaledparasOG)

    varshiftdel = GrAl.bayespriorshift(cutvarhold, meandel)

    varshiftdel = var.append(varshiftdel)


    """NEED SOMETHING HERE TO FIX SCALED AND FUNC PARAS"""
    varshiftdel = varshiftdel.sort_values(4)
    mean = []
    for i in range(varshiftdel.shape[0]):
        code = varshiftdel.iloc[i][4]
        if code == 35:
            #cutvarhold.loc[cutvarhold.index[i], 5] = x
            rempara.append(i)
        #    meandel.append(0.5)
        mean.append(varshiftdel.iloc[i][0])

    #this removes the nameing convention
    dic = {}
    for i in range(varshiftdel.shape[0]):
        dic.update({varshiftdel.index[i]: i})
    dfnew = varshiftdel.rename(index=dic)

    print(rempara)
    for value in rempara[::-1]:
        scaledparasOG,functparasOG,funcvar_holder, varshiftdel, data,cutdic = GrAl.listmodifier(value,scaledparasOG,functparasOG,funcvar_holderOG,varshiftdel,data,mean,Mspaces)

    print(scaledparasOG)
    print(functparasOG)
    print(varshiftdel)

    # update mecset with these values
    MEC_set.update({"var": varshiftdel, "funcvar": functparasOG, "scalvar": scaledparasOG})
    inputwritter(filenametot + '/ADMCMCallparas_output.txt', "ADMCMC", logicm, data, expfile,varshiftdel, winfunc, settingsdic,
                 MEC_set, trunstime)


print("finished")

