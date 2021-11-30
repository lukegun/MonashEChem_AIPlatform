

import time
import datetime
import os
import numpy as np
import pandas as pd
import Script_generator as Scripgen
import MECwrit_mod as MECr
import window_test as Wint
import plotting_scripts as plotter
import MultiharmVsV_mod as MHplot
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.fftpack import rfftfreq
import sys

input_name = sys.argv[1]
serverdata, inputMEC, ReactionMech, RedoxOrder, AC_case, AutoNharm, Cnoise, N, seperating, decilist, samplertype,\
windowing, guass_std, bandwidth, varibles, scalvar, funcvar, nondim, trun = MECr.inputloader(input_name)

# gaussian distrabuted noise
#Cnoise = 0.0
sqlExpinputlist = [False, 1]  # EXPsetrow and wiether no to put the values into the array [ True = use previous exp input; False = input expsettings, #EXPsetrow  ]

starttime = time.time()
"""input types"""
# for gaussian x = [mean,std,input,repeat]
# for standard dev x = x = [resultion,low,high,log scale,input,repeat]


data = MECr.mecreader(inputMEC)

spaces = Scripgen.data_finder(data)

# Turns varibles into a usable thing
var = []
for i in range(len(varibles)):
    # for gaussian x = [mean,std,input,repeat]
    # for standard dev x = x = [resultion,low,high,log scale,input,repeat]
    """Addscan type in here"""
    if samplertype == 'normaldist':
        x =[varibles[i][0],varibles[i][1],None,0,varibles[i][2],varibles[i][3]]
    elif samplertype == 'linear':
        x = [varibles[i][0], varibles[i][1], varibles[i][2], varibles[i][3], varibles[i][4], varibles[i][5]]
    elif samplertype == 'unirand':
        # min value, max value, value,repeat, log
        x = [None, varibles[i][0], varibles[i][1], varibles[i][2], varibles[i][3], varibles[i][4]]
    else:
        print('wrong sampler type inputed')
        exit()

    var.append(x)

var = pd.DataFrame(var, columns=None)

# parameter
#space_holder = var[[4, 5]].values  # retains values for use in printed values in np.array # retains values for use in printed values in np.array
var, scalvar = Scripgen.line_assigner(var, spaces, scalvar)
cap_series = MECr.capacatance0(data,spaces,var)
AC_freq, AC_amp = MECr.AC_info(data, spaces)
DCAC_method = MECr.ACDC_method(AC_amp)
funcvar_holder = MECr.functionholder(var)  # issolates the functional parameters
freqvar = Scripgen.freq_test(var) # gets the

# sets up args
if samplertype == 'normaldist':
    inmeans = []
    instd = []
    for i in range(len(varibles)):
        inmeans.append(varibles[i][0])
        instd.append(varibles[i][1])
    samples = []
    for i in range(N):
        x = np.random.normal(inmeans,instd)
        samples.append(x)


elif samplertype == 'linear':

    samples = []
    print('oh fuck your going to have a real bad scaling problem on this dimensional thing')

#unpacks var propiies for ease of use
elif samplertype == 'unirand':      # this is to work around for the scaling issues

    lowerbound = []
    upperbound = []
    meanbound = []
    logstyle = []

    for i in range(len(varibles)):
        lowerbound.append(varibles[i][0])
        upperbound.append(varibles[i][1])

        logstyle.append(varibles[i][2])
        if varibles[i][2] == 0: # no log sensiive
            meanbound.append((varibles[i][1]+varibles[i][0])/2)
        elif varibles[i][2] == 1: # logsensitive
            meanbound.append(10**((np.log10(varibles[i][1]) + np.log10(varibles[i][0])) / 2))
        else:
            print("incorrent log parameters")
            exit()


"""NEED SOMETHING HERE TO POOL TO 4 different processes and Run the total samples in batches of 20"""
"""WILL NEED TO """

#MECSim constants
# get time and voltage for simulation
"""CAN BE REMOVED IN FUTURE UPDATES"""
Timesim, voltagesim = MECr.outputsetter(data,AC_freq, AC_amp)

Nsimlength = 2**int(data.iloc[6]) # Simulation legnth
HarmMax = 9 #Max harmonics where going to consider
nsimdeci,frequens = MECr.Simerharmtunc(Nsimlength, Timesim[1], bandwidth,HarmMax,AC_freq)

# sets up the filter windows
filter_hold = []
fft_freq = rfftfreq(Nsimlength, d=Timesim[1])
if windowing == "Convolutional_Guass":

    Convguass = Wint.analitical_RGguassconv_fund(fft_freq, bandwidth[1][1], guass_std * bandwidth[1][1])
    filter_hold.append(Convguass)

    for NNNN in range(1, 13):
        # analytical solution to guass rec convultion
        Convguass = Wint.analitical_RGguassconv(fft_freq, bandwidth[1][1], NNNN * AC_freq[0], guass_std * bandwidth[1][1])
        filter_hold.append(Convguass)

elif windowing == "Rect":

    rect = Wint.square_window_fund(fft_freq, bandwidth[1][1])
    filter_hold.append(rect)

    for NNNN in range(1, 13):
        rect = Wint.square_window(fft_freq, NNNN * AC_freq[0], bandwidth[1][1])
        filter_hold.append(rect)

else:
    print("Error: no correct filter windowing inputted")
    exit()

# If false is puts the settings into he input
if not sqlExpinputlist[0]:
    with open(input_name) as f:
        inputtext = f.read()
    sqlExpinput = MECr.sqlexpsettings(serverdata,data, spaces, nondim, AutoNharm,bandwidth,ReactionMech,
                                      windowing,guass_std,inputtext)
else:
    sqlExpinput = sqlExpinputlist[1]

MEC_set = {'data': data, 'var': var, 'spaces': spaces, 'DCAC_method': DCAC_method,
                   'scalvar': scalvar, 'funcvar': funcvar, 'funcvar_holder': funcvar_holder,
                   'cap_series': cap_series, 'AC_freq':AC_freq, "AC_amp":AC_amp,'deltatime':Timesim[1],
                   'bandwidth':bandwidth,"nsimdeci":nsimdeci,"RedoxOrder":RedoxOrder,"ReactionMech":ReactionMech,
                    "decilist":decilist, "sqlExpinput":sqlExpinput, "serverdata":serverdata,"AutoNharm":AutoNharm,
                    "filter_hold":filter_hold, "trunvec":trun, "AC_case":AC_case, "freqvar":freqvar, "windowing":windowing,
                    "guass_std":guass_std, "nondim":nondim}

# Attaches kwargs to wrapper function this is a wrapper -> Iterative_MECSim0

Mec_sql = MECr.MECSim_SQL(**MEC_set)
model = Mec_sql.sql_loopr

#Runs till the overall function is ccomplete

# preallocation of points for reaction ID
intilReaction_ID = MECr.sqlReacIDPreallo(serverdata,N,sqlExpinput,ReactionMech)
RIDcount = 0 # counter for each reaction ID
log = []        # saves all he values for the output log

for i in range(int(np.ceil(N/seperating[0]))+1):
    # generate random uniform samples
    samples = []
    for i in range(seperating[0]):
        x = np.random.uniform(-1, 1,len(varibles))
        # some function to convert uniform to useable parameters
        xp = MECr.scaler(x,lowerbound,upperbound,meanbound,logstyle)
        xp = np.concatenate((xp,np.array([intilReaction_ID + RIDcount]))) # this preallocates the varibles to specifc thing
        RIDcount += 1
        samples.append(xp)
        log.append(xp)

    # for testing purposes
    #model(xp)
    #exit()

    # Multiprocessing call around Iterative MECSim only doing 4 at a time
    # preallocate all the sql rows then put data into the randomly generated data
    with Pool(processes=seperating[1]) as p:
        multi = p.map(model, samples)

"""# make a file for reaction mechanisms  but check one isnt there to begin with
file = True
i = 0
while file:
    try:
        if i == 0:
            filename = "Output_"+ ReactionMech +"_"+ str(datetime.date.today())
            os.makedirs(filename)
        else:
            filename = "Output"+str(i)+"_" + ReactionMech +"_"+ str(datetime.date.today())
            os.makedirs(filename)
        file = False
    except:
        print("file already exists")
    finally:
        i += 1"""

#print("save React_ID : paras to files called Eset.txt")
"""f = open(filename+ "/log.txt",'w+')
N = len(log[0]) - 1
for input in log:
    f.write(str(input[-1]) + "\t")
    for i in range(N-1):
        f.write(str(input[i]) + "\t")
    f.write(str(input[N-1]) + "\n")
f.close()"""

#print("Do one D plot histograms of paras tried")
"""# this is for naming of varibles
nametest = []
for x in varibles:
    nametest.append([x[3],x[4]])

name = plotter.name_Allo(nametest)"""

"""i = 0
for title in name:
    # extracts the parameters from log into user friendly system due to np array
    x = []
    for trails in log:
        x.append(trails[i])

    if varibles[i][2] == 0: # check for log sensiivity
        plt.figure()
        plt.hist(x)
        plt.xlabel(title)
        plt.savefig(filename + "/Hist_V" +str(nametest[i][0]) + "-R"+str(varibles[i][4]))
        plt.close()
    else:
        plt.figure()
        plt.hist(np.log10(x))
        plt.xlabel("Log10 " + title)
        plt.savefig(filename + "/Hist_V" + str(nametest[i][0]) + "-R" + str(varibles[i][4]))
        plt.close()

    i += 1

f = open(filename + "/dump.txt" ,"w")
f.write("Varibles region: \n")
for varreg in varibles:
    for x in varreg:
        f.write(str(x)+ ", ")
    f.write("\n")
f.write("\nScalvar:\n")

for varreg in scalvar:
    for x in varreg:
        f.write(str(x)+ ", ")
    f.write("\n")
f.write("\nfuncvar\n")

for varreg in funcvar:
    for x in varreg:
        f.write(str(x)+ ", ")
    f.write("\n")
f.write("\nfuncvar\n")"""

comptime = (time.time() - starttime)/60

print("Finished\nCompletion Time (mins): " + str(comptime))
