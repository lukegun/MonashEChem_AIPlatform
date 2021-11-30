
import numpy as np
import matplotlib.pyplot as plt
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
    print(Nharm)
    bandwidthplot = np.array(bandwidth)
    bandwidthplot = bandwidthplot[:,:Nharm]


    # generate filters
    simfilterhold = wint.RGguassconv_filters(2**Nsim, bandwidth, timedata[1]/(2**Nsim), [ACsettings[0]], std)
    expfilterhold = wint.RGguassconv_filters(len(currenthold[0]), bandwidth, timedata[0], [ACsettings[0]], std)

    # generate all harmonics
    hil_storehold = []
    fundharmhold = []
    for current in currenthold:
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
        harm = np.average(harms,axis= 0)
        xy.append(harm)
    hilstore = xy

    percentagetrun = 0.05
    truntime = percentagetrun*timedata[1]

    trunvec_sim, trunvec_exp = gm.EXPharmtreatment(len(hilstore[0]), timedata[1], [truntime,  timedata[1] - truntime ], 2**Nsim)
    print("wooo")
    print(trunvec_sim)
    #stored for the capacitance fitting before truncation
    capcurrent = hilstore[1]

    xy = []
    for hils in hil_store:
        harm = wint.Average_noiseflattern(hils, trunvec_exp, avgnum=20)
        xy.append(harm)
    current = xy

    "truncate edges"

    del  hil_storehold

else: # DC mode
    current = np.average(currenthold,axis= 0)

    # stored for the capacitance fitting before truncation
    capcurrent = current
    del currenthold # clear up ram
    fundimentalharm = None # required due to ac process
    ACsettings = None
# majority of the above can be saved to the training DNN then loaded from

"""for i in range(Nharm):
    # check issues with the harmonics but theres a couple of stuff
    plt.figure()
    plt.plot(hilstore[i])
    plt.savefig("harmtest"+str(i)+".png")
    """#"""INsert into the CMAOPT stuff put into biomec""""""


""" Fit double layer capacitance automatically """
capfitclass = cap.overallcapfit(capcurrent,ACmode,trunvec_exp,timedata[0],Evalues,scanrate,expvoltage,fundimentalharm,ACsettings)
capparas, Icdlfit, E0values = capfitclass.overall_fitting() # gets the cap parameter values, background charging current and E0values


# correction from farads to farads /cm**2
invArea = 1/paraestdic.get(6)
for i in range(len(capparas) - 1):
    capparas[i] = capparas[i]*invArea

# collects all possible parameters to be fit for the reaction mechanism
allpara,scaledparas,functparas,Mspaces,Nreact = gm.Reactparaallocator(reactmech[0])

""" Save all of above to a txt file """

# below gets data from the sql database
""" Load a output from another program PROBS BOT NEEDED graded trained subsets"""

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


"""NEED SOMETHING HERE SO we can identify if its reduction or oxidation"""
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

#cuts the bandwidth down to size
for i in range(len(bandwidth)):
    bandwidth[i] = bandwidth[i][:Nharm] # cuts nandwidth down to a useable length


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
op_settings[3] = float(0.05)  # sets the tolx value might increase to 0.1 if too slow
op_settings[4] = float(0.33)  # sets the initial sigma

sigma = None # uses this as well use HarmPer # can always use Bayes_expharmper

header = ["STANDARD", "HarmPerFit", "CMAES"]

# changes to scalar func and func to format
x =[[len(scaledparas)]]
for i in scaledparas:
    x.append(np.array(i))
scaledparas = x

#make the last input in funct and scaled change to the data row FIX
print(scaledparas)
print(functparas)
if len(functparas) != 0: # this is for some standardisation
    x =[functparas[0]]
    for i in functparas[1:]: #skip the initial value
        x.append(np.array(i))
    functparas = x
else:
    functparas = [[0]]


#functparas.insert(0,[len(functparas)])

# sets up AC settings in the old format
AC_freq = np.array([ACsettings[0]]) # just set up for one atm
AC_amp = np.array([ACsettings[0]]) # just set up for one atm

DCAC_method = gm.ACDC_method(AC_amp, op_settings)

# issolates the functional varible
funcvar_holder = gm.functionholder(var)

# sort the below out
scaledparas = gm.scaledline_assigner(Mspaces, scaledparas)

print(current)
current= np.array(current)
plotcurrent = current # here for plotting stuff
print(current.shape)
print(truntime)
trunstime = [truntime,timedata[-1]-truntime]
deci = int(len(current[0,:])/op_settings[2])
current = current[:,::int(deci)]
current = current[:,trunvec_sim[0]:trunvec_sim[1]]

# this is here to remove pinned varibles and correct all the other varible lines done in varallocator
data1 = data
for value in priorsdel:
    mean = var.iloc[:][0].values # THIS WORKS BUT ITS GENERALLY TO AVOID REDOING STUFF
    scaledparas,functparas,funcvar_holder,var,data = GrAl.listmodifier(value, scaledparas, functparas, funcvar_holder, var,data,mean)
print(data == data1)
exit()
# Results are the best fit and logtot are all trails tried
#Results are in format [-1,1]

"""WE USE CMA ES due to previous use, a multimodalular bee algorithm would probably work better but would be more data hungry and require more CPUs"""

timetot = time.time()
itteration = 0
stdhold = []
meanhold = []
varsave = []
complete = False

while not complete:

    freq_set = {'harm_weights': harm_weights, 'bandwidth': bandwidth}  # here to allow functional ACDC functionality

    MEC_set = {'data': data, 'var': var, 'spaces': Mspaces, 'DCAC_method': DCAC_method, 'Exp_data': current,
               'scalvar': scaledparas, 'funcvar': functparas, 'funcvar_holder': funcvar_holder,
               'cap_series': cap_series, 'op_settings': op_settings, 'Nsimdeci': trunvec_sim,
               'harm_weights': harm_weights, 'bandwidth': bandwidth, 'Exsigma': sigma, 'AC_freq': AC_freq,
               'filters': simfilterhold}

    print(MEC_set)
    """
    # Fix this to make it most optimal
    Nd = 8 # number of dimensions
    Nfunc = int(4 + 3 * np.log(Nd)) # this is number of functions per CMA
    print("fucck")
    if itteration == 0:
        Niter = 3
        rawarray,array, results,meanerror = GrAl.convergencetest(var, op_settings, header, MEC_set, current,Niter)

        
        rawstd = []
        std = []
        mean = []
        i = 0
        for values in array:
            m = np.mean(values)
            mean.append(m)  # can check if something has converged
            RSD = np.std(values) / abs(m)
            rawstd.append(np.std(rawarray[i]))
            if RSD >= 1: # case where it flys of to infinite
                std.append(1)
            else:
                std.append(RSD)
            i += 1

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
            m = np.mean(values)
            mean.append(m)  # can check if something has converged
            RSD = np.std(values) / abs(m)
            rawstd.append(np.std(rawarray[i]))
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
    print("Percentage errror: ", end="")
    print(meanerror)
    print(var)
    print(std)
    print(rawstd)
    print(mean)"""

    std = [0.05210914047442646, 0.017914130401366215, 6.650513595752767e-05, 0.0787944159232518, 0.08762990423674988, 0.06916430147747195]
    rawstd = [0.012062243382699022, 0.056029649878264164, 0.0007894287139620328, 0.7891838883465675, 0.012613227846990697, 0.46114844396719407]
    mean = [41.337793978371394, 6.255358046739589e-06, 0.35610575150122986, 0.5007866858961534, 0.01576265571233239, 83.33959647153267]


    #plotting stuff is here for the diagnostics
    # converts to accutual parametrs
    #var_out, Perr = standCMA.CMA_output([mean], **MEC_set)

    #print(var_out, Perr)

    #Scurr = MLsp.Iterative_MECSim_Curr(*[mean], **MEC_set)

    # gets a linear time array
    #Exptimearray = np.linspace(0, timedata[-1], len(plotcurrent[0, :]))

    #Simtimearray = np.linspace(0, timedata[-1], len(Scurr))

    #plotter.harmplot('harmonicplots_I'+str(itteration), Scurr, Simtimearray, plotcurrent, Exptimearray, bandwidthplot, AC_freq, Mspaces)


    #Make an exception here so if std is avoe this amount skip everthing and pin those parameters
    #If standard deviation hasn't been meet  build a bunch of varibles of the trees

    # Scaning over a uniform distrabution between -1 and 1
    # A completely uninformative distrabution would have std sqrt(4/12) aprox =  0.6
    # A half uninformative distrabution would have std sqrt(1/12) aprox = 0.3
    # as such 0.4 was set to be mostly uninformative enough parameter is mostly random
    maxstdvalue = 0.4 # make a loaded in parameter (parameter is varied between -1 and 1)
    maxRSD = 0.2 #

    logsensitive = [i for i in var.iloc[:][3].values]  # etracts is the log sensitivity of parameters




    # below fails

    for i in range(len(std)):
        if logsensitive[i] == 0: # not log sensitive system
            std[i] = 0
        else: # log sensitive system
            rawstd[i] = 0
    print("corrected stds")
    print(logsensitive)
    print(mean)
    print(std)
    print(rawstd)
    maxacceptpara = []


    #boolrawstd, boolRSD
    boolrawstd = [i > maxstdvalue for i in rawstd]
    boolRSD = [i > maxRSD for i in std]
    print(boolRSD)
    print(boolrawstd)
    Nstd = sum(boolrawstd) + sum(boolRSD)

    """SOMETHING TO SET THE PINNED PARAMETER VALUES AS MEAN OF ABOVE STD"""

    """TEST THE EC OPTIMISATION"""

    """ALSO NEED SOMETHING TO KILL IT IF IT FAILS BIG TRY statement but seems gross"""

    """NEED AN EXCEPTION HERE FOR LOG DIST STD based stuff: USE RSD IF PARAMETR IS IN LOG SCALE SHOULD BE DONE"""

    """UPDATE MEC_Set"""
    #PUT THE BELOW IN HERE
    """ NEED SOMETHING TO PUT IN A BETTER APPROXIMATION OF THE PINNED VALUE BASED ON THE CALCULATED MEAN (depends on what parameter)"""

    """SOMETHING TO ADJUST THE PRIOR if means are close to -+ 1"""

    # in case therre isnn't many parameters left it just pins the worst
    if var.shape[0] == 5 and (Nstd != 1 or Nstd != 0):
        Nstd = 2
    elif var.shape[0] <= 4 and Nstd != 0:
        Nstd = 1

    if Nstd == 0:
        if any(jj > 0.1 for jj in rawstd) or any(jj > 0.05 for jj in std):
            greedfit, var, scaledparas, functparas, funcvar_holder,data = GrAl.greedypin(var, scaledparas, functparas,
                                                                                    funcvar_holder,op_settings, header,
                                                                                    MEC_set, current,rawstd,
                                                                                    std,boolrawstd, boolRSD,mean)
            """need another thing for pinning to approx mean"""
        else: # greedy algorithm has succseded
            complete = True

    elif Nstd == 1 or Nstd == 2: #if its only bad for some just pin those
        stdind,data = GrAl.stdallocator(rawstd,std, Nstd,boolrawstd, boolRSD,mean,var,Mspaces,MEC_set.get("data"))
        for values in stdind:
            scaledparas, functparas, funcvar_holder, var,data = GrAl.listmodifier(values,scaledparas,functparas,funcvar_holder,var,data,mean)
            print("done")
            print(functparas)

        # Need to do a single optimisation here to pass to above
        # update the MEC_set dictionary
        MEC_set.update({'data': data})
        MEC_set.update({'var': var})
        MEC_set.update({'scalvar': scaledparas})
        MEC_set.update({'funcvar': functparas})
        MEC_set.update({'funcvar_holder': funcvar_holder})
        print(scaledparas, functparas, funcvar_holder, var)
        # optimisation loop
        results, logtot = standCMA.STAND_CMAES_TOTCURR(var, op_settings, header[1], MEC_set, current)
        rawgreedfit = results[0]


    else: # if std is bad skip below step and just pin worst 3
        stdind,data = GrAl.stdallocator(rawstd,std, 3,boolrawstd, boolRSD,mean,var,Mspaces,MEC_set.get("data"))

        print(data)
        exit()

        for values in stdind:
            scaledparas, functparas, funcvar_holder, var,data = GrAl.listmodifier(values, scaledparas, functparas,
                                                                             funcvar_holder, var,data)
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

        # Need to do a single optimisation here to pass to above

    # gets the dimensionalised values and error
    greedfit, greederror = standCMA.CMA_output([rawgreedfit], **MEC_set)

    # exception if we run out of parameters to pin and test
    if var.shape[0] == 1 or var.shape[0] == 0: # zero is real bad but may occur
        complete = True

    itteration += 1 # counts the number of iteration layers

    # Might want something here to plot the convergence of the parameters

print("Greedy Algo Time: " + str((time.time()-timetot)/60))

print("results")
print(varsave)
print(stdhold)
print(meanhold)

"""ADD SOMETHING HERE TO DO A FINAL HIGH PRECISSION OPTIMISATION USING THE PINNED PARAMETERS"""

# converts to accutual parametrs
var_out, Perr = standCMA.CMA_output(results, **MEC_set)

print(var_out, Perr)

Scurr = MLsp.Iterative_MECSim_Curr(*[var_out], **MEC_set)

# gets a linear time array
Exptimearray = np.linspace(0,timedata[-1],len(plotcurrent[0,:]))

Simtimearray = np.linspace(0,timedata[-1],len(Scurr))

plotter.harmplot('harmonicplots', Scurr, Simtimearray, plotcurrent, Exptimearray, bandwidthplot, AC_freq, Mspaces)
#completion_time = (time.time() - t1) / 60

# complete the greedy algorithm

""" Greedy algorithm stuff """


""" Print to a final output """


""" Set up CMA ES input """
#input loader modified CMAES

""" Set up BI input file """
#input loader modified BI