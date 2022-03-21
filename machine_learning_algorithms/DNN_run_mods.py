import string
import numpy as np
from itertools import islice
from pandas import read_csv
import numpy as np
from scipy.fftpack import rfft, irfft, rfftfreq
from scipy.signal import hilbert as anal_hil_tran # NEED signal to make a smooth envolope
import matplotlib.pyplot as plt

# extract stuff from the terminal line
def terminalextract(terminal):
    inputsettings = terminal
    # removes the file name
    inputsettings.remove(inputsettings[0])
    print(inputsettings)

    expfile = []
    i = 0
    while inputsettings[i][0] != "-":
        expfile.append(inputsettings[i])
        i += 1

    inputdic = {}
    x = inputsettings[i]
    odd = []
    i += 1
    while inputsettings[i][0] != "-":
        odd.append(inputsettings[i])
        i += 1
    inputdic.update({x: odd})

    x = inputsettings[i]
    odd = []
    i += 1
    while i != len(inputsettings):
        odd.append(inputsettings[i])
        i += 1
    inputdic.update({x: odd})

    DNNarch = inputdic.get("-dnn")
    parameterestimates = inputdic.get("-p")

    return expfile,DNNarch,parameterestimates

def paraextract(parameterestimates):
    paradic = {}
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase

    # this is for scientific notation
    lowercase = lowercase.replace("e", "")
    uppercase = uppercase.replace("E","")
    strings = lowercase+ uppercase
    for values in parameterestimates:
        x = values.split("=")
        y = letters2values(x[0])
        s = ""
        i = 0
        for letters in x[1]:
            if letters not in strings:
                s += letters
            else:
                break
            i += 1

        unit = ""
        for j in range(i,len(x[1])):
            unit += x[1][j]

        """Something to convert units into MECSim friendly units"""
        s = unitconversion(y,float(s),unit)
        print(y,s,unit)
        paradic.update({y:s})


    return paradic

def letters2values(x):

    if x == "C": # Concentration
        x = 21
    elif x == "D": # diffusion
        x = 22
    elif x == "A":  # electrode area
        x = 6
    elif x == "alp":  # electrontransfer rate
        x = 35
    elif x == "k0": # electrontransfer rate
        x = 34
    elif x == "E0": # electrontransfer rate
        x = 33
    elif x == "Ru": # electrontransfer rate
        x = 11
    elif x == "T":   # Temp
        x = 1
    else:
        print("ERR: Incorrect parameter estimates")
        exit()

    return x

def unitconversion(code,value,unit):

    # this is just to make sure there arn't small issues
    unit = unit.lower()

    if code == 21: # concentration
        if unit == "uM".lower(): #micromolar
            value = value * 10 ** (-9)
        elif unit == "mM".lower():
            value = value*10**(-6)
        elif unit == "M".lower():
            value = value * 10 ** (-3)
        elif unit == "mol/cm3":
            value = value

        elif unit == "mol/cm2": # surface concentartion
            value = value
        else:
            pass

    elif code == 22: # Diffusion

        if unit == "cm2/s":
            value = value
        elif unit == "m2/s":
            value = value * 10000
        else:
            pass

    elif code == 6: # Electrode area
        if unit == "cm2":
            value = value
        elif unit == "mm2":
            value = value*0.01
        elif unit == "m2":

            value = value * 10000
        else:
            pass

    elif code == 35: # alpha
        #unitless
        pass

    elif code == 34: # k0
        # not much to do here
        pass

    elif code == 33: # potential
        if unit == "mV".lower():
            value = value* 10**3
        elif unit == "V".lower():
            value = value
        elif unit == "uV".lower():
            value = value * 10 ** 6
        else:
            pass
    elif code == 1: #temp
        if unit == "c".lower():
            value = value + 273.15
        elif unit == "K".lower():
            value = value
        elif unit == "f":
            value = (value - 32) * 5 / 9 + 273.15

        else:
            pass

    elif code == 11: # Ru
        if unit == "ohm" or unit == "ohms":
            value = value
        elif unit == "kohm" or unit == "kohms":
            value = value*1000
        else:
            pass

    else:
        pass

    return value


# nondimensionalisation of current
def currentnondimsurfconfined(Scurr,area, surfaceconc, scanrate, temp):

    F = 9.648670*10**3 #emus/mole
    R = 8.31446261815324*10**7 #erg⋅K−1⋅mol−1
    v = scanrate*0.0033356405 #statV per second

    e0 = R*temp/F  #statV

    T0 = e0/v

    constant = T0/(F*area*surfaceconc*10) # time 10 for the current to biot
    scurr = Scurr*constant

    return scurr, constant


def currentnondim(Scurr,area, diffusion, conc):

    F = 9.648670 * 10**3 #emus/mole
    r = np.sqrt(area/np.pi)
    constant = 1/(np.pi*r*F*diffusion*conc*10) # time 10 for the current to biot
    scurr = Scurr*constant

    return scurr, constant

def normalisationrange(currenttot,constant):
    transformedI = []
    first = True
    x = [-1, 1]
    for current in currenttot:
        print("testing without averaging")
        #y = (x[1]-x[0])*((current-np.min(current))/(np.max(current)-np.min(current))+ x[0])
        y = current
        transformedI.append(constant*y)
        if first:
            x = [0, 1]
            first = False

    return transformedI


    return transformedI

# loads in a file of generic parameter values
def settingsload(filename):

    with open(filename) as f:
        file = f.readlines()

    numiter = int(file[0].split("=")[1])

    stdaccept = float(file[1].split("=")[1])

    return numiter, stdaccept

#file to load the exp data
def expextract(filename):

    # loads the first few lines of input file to memory
    with open(filename, 'r') as file:
        lines_gen = islice(file, 48)
        linesstore = []
        for lines in lines_gen:
            s = lines.strip("\n")
            linesstore.append(s)

    # all the below is for cvsin_type_1 format
    if linesstore[0] == "cvsin_type_1":
        Estart = float(linesstore[1].split("\t")[1])
        Eend = float(linesstore[2].split("\t")[1])
        Emax = float(linesstore[3].split("\t")[1])
        Emin = float(linesstore[4].split("\t")[1])

        Ncycles = float(linesstore[10].split("\t")[1])

        Evalues = [Estart,Eend,Emax,Emin,Ncycles]

        freq = float(linesstore[5].split("\t")[1])
        amp = float(linesstore[6].split("\t")[1])

        ACsettings = [freq,amp]

        scanrate = float(linesstore[8].split("\t")[1])

        results = read_csv(filename, delim_whitespace=True, skiprows=19, names=["v", "i", "t"])
        current = results.values[:,1]  # changes results fro df object to np array and extract current
        voltage = results.values[:,0]

        Dt = results.values[1,2]
        Tmax = results.values[-1,2]
        timedata = [Dt,Tmax]
    else:
        print("ERROR: INCORRECT EXPERIMENTAL FILE EXIT: in FUNCTION expextract")
        exit()

    return Evalues, ACsettings,scanrate,current,voltage,timedata


#extracts windows for log10 auto counter
def Simerharmtunc(Nsimlength, exptime, bandwidth,HarmMax,AC_freq):
    # this checky fix migt cause some issues
    Nex = [0]
    truntime = [0, HarmMax*AC_freq[0]+2*AC_freq[0]]
    frequens = rfftfreq(Nsimlength, d=exptime)

    N = HarmMax #len(bandwidth[0])
    nsimdeci = []

    # DC section
    if bandwidth[0][0] != 0:
        if truntime[0] < bandwidth[0][0]:
            Nhigh = -Nex[0] + find_nearest(frequens, bandwidth[0][0])
            nsimdeci.append([0,Nhigh])

    j = 0
    while j != 1:
        i = 0
        while i != N:
            if (i+1)*AC_freq[j] + bandwidth[j + 1][i]/2 > truntime[1] or (i+1)*AC_freq[j] + bandwidth[j + 1][i]/2 >max(frequens):
                if (i + 1) * AC_freq[j] - bandwidth[j + 1][i]/2 < truntime[1]:
                    Nwindlow = -Nex[0] + find_nearest(frequens, (i + 1) * AC_freq[j] - bandwidth[j + 1][i]/2)
                    Nwindhigh = Nsimlength  # this will probs be some arbitary number/freq
                    #freqx.append(frequens[Nwindlow:Nwindhigh])
                    nsimdeci.append([Nwindlow, Nwindhigh])
                i = N
            else:
                Nwindlow = -Nex[0] + find_nearest(frequens, (i+1)*AC_freq[j] - bandwidth[j + 1][i]/2)
                Nwindhigh = -Nex[0] + find_nearest(frequens, (i+1)*AC_freq[j] + bandwidth[j + 1][i]/2)
                #freqx.append(frequens[Nwindlow:Nwindhigh])
                nsimdeci.append([Nwindlow,Nwindhigh])
                i += 1
        j += 1

    return nsimdeci,frequens

# function for finding the nearest value in a series of array
def find_nearest(array, value):
    ''' Find nearest value is an array '''
    idx = (np.abs(array-value)).argmin()
    return idx

# counts the number of harmonics in experimental data
def harmoniccounter(Curr,nsimdeci,AutoNharm):

    fft_res = rfft(Curr)

    # this can be done seperatly So lok into it
    #freq = rfftfreq(len(fft_res),d = exptime) try yo pass through DELTAexptime
    powerspec = np.log10(np.abs(fft_res) / len(fft_res))
    """Will need the code to extract harmonics and background"""

    PSBG, PSBGlist = PSbackextract(powerspec, nsimdeci)             #extracted background
    PSsignal, PBsiglist = PSsigextract(powerspec, nsimdeci)          #extracted signals

    """plt.figure()
    plt.plot(PSsignal)
    plt.plot(PSBG)
    plt.savefig("FUCKER.png")
    plt.close()"""

    #checks first harmonic to Harmax
    for i in range(1,len(PSBGlist) - 1):

        PSdiff = AutoNharm     #

        # Calculates the average max of the background
        try:
            backaverage = (max(PSBGlist[i])+max(PSBGlist[i+1]))/2
        except:
            print("ERROR in harmoniccounter function")
            print(PSBGlist)
            print(nsimdeci)
            plt.figure()
            plt.plot(PSsignal)
            plt.plot(PSBG)
            plt.savefig("FUCKER.png")
            plt.close()
            print("weez")
            print(i)
            print(len(PSBGlist))
            Error
            exit()

        sigmax = max(PBsiglist[i])
        if sigmax < backaverage + PSdiff:
            break

    Nharm = i -1

    return Nharm, fft_res

def PSbackextract(powerspec, nsimdeci):

    PSBG = np.empty([0])  # Scurr[Nsimdeci[0][0]:Nsimdeci[0][1]]
    PSBGlist = []
    i = 0
    for X in nsimdeci[0:-1]:
        PSBG = np.concatenate((PSBG, powerspec[X[1]:nsimdeci[i + 1][0]]))
        PSBGlist.append(powerspec[X[1]:nsimdeci[i + 1][0]])
        i += 1

    return PSBG, PSBGlist


def PSsigextract(powerspec, nsimdeci):

    PSsignal = np.empty([0])  # Scurr[Nsimdeci[0][0]:Nsimdeci[0][1]]
    axis = np.empty([0])  # Scurr[Nsimdeci[0][0]:Nsimdeci[0][1]]
    PSsiglist = []
    for X in nsimdeci:
        PSsignal = np.concatenate((PSsignal, powerspec[X[0]:X[1]]))
        PSsiglist.append(powerspec[X[0]:X[1]])

    return PSsignal, PSsiglist

# harmonic generator
# generates harmonics envolopes from MECSim current output
def Harmonic_gen(fft_res, filters):  # cuts the harmonic fourier space data out {NEEDS TO EXCLUDE FUNDEMENTAL}

    nharm = len(filters)  # counts the number of harmonics
    # Np = len(fft_res)    # gets the number of datapoints
    # harmstore = np.zeros(nharm*spaces[4] ,int(Mwin*(2/g)))
    hil_store = []
    # N_ifft = np.zeros((nharm*Nac + 1))

    # hil_store[0,:] = irfft(y)  # generates fundimental harmonics from cut window

    x = fft_res * filters[0]  # This need to be truncated in the future

    harmonic = irfft(x)  # generates harmonics

    hil_store.append(harmonic)
    i = 1
    while i != nharm:
        x = fft_res * filters[i]  # This need to be truncated in the future

        harmonic = irfft(x)  # generates harmonics
        #pops out the fundimental harmonic for cap
        if i == 1:
            fundimentalharm = harmonic

        hil_store.append(abs(anal_hil_tran(harmonic)))  # uses HILBERT TRANSFORM to generate the envolope

        # using the abs fixed an issue with the complexes disapearing in the return
        i += 1

    return hil_store,fundimentalharm

# truncates and passes around experimental harmonics and find truncation points
def EXPharmtreatment(nEX,Extime, truntime, Nsim):

    E1 = find_nearest(np.linspace(0, Extime, nEX), truntime[0])
    Ndeci = Nsim

    E1 = int(E1 - E1 % (nEX / Ndeci))  # sets a point for E1 that has Int in simulation file
    Esim1 = E1 * Ndeci / nEX

    if truntime[1] == 'MAX':
        E2 = nEX
        Esim2 = Ndeci
    else:
        E2 = find_nearest(np.linspace(0, Extime, nEX), truntime[1])
        E2 = int(E2 + ((nEX / Ndeci) - E2 % (nEX / Ndeci)))

        Esim2 = E2 * Ndeci / nEX

    Nsimdeci = [int(Esim1), int(Esim2)]
    Nex = [E1, E2]

    # truncates experimental harmonics
    #EX_hil_store = EX_hil_store[:, E1:E2]
    # need to identify correct points for simulation and experimental so that its the same time spot then pass identified decimated simulation values to MEC_set

    return Nsimdeci, Nex

def expextract(filename):

    # loads the first few lines of input file to memory
    with open(filename, 'r') as file:
        lines_gen = islice(file, 48)
        linesstore = []
        for lines in lines_gen:
            s = lines.strip("\n")
            linesstore.append(s)

    # all the below is for cvsin_type_1 format
    if linesstore[0] == "cvsin_type_1":
        Estart = float(linesstore[1].split("\t")[1])
        Eend = float(linesstore[2].split("\t")[1])
        Emax = float(linesstore[3].split("\t")[1])
        Emin = float(linesstore[4].split("\t")[1])

        Ncycles = float(linesstore[10].split("\t")[1])

        Evalues = [Estart,Eend,Emax,Emin,Ncycles]

        freq = float(linesstore[5].split("\t")[1])
        amp = float(linesstore[6].split("\t")[1])

        ACsettings = [freq,amp]

        scanrate = float(linesstore[8].split("\t")[1])

        results = read_csv(filename, delim_whitespace=True, skiprows=19, names=["v", "i", "t"])
        current = results.values[:,1]  # changes results fro df object to np array and extract current
        voltage = results.values[:,0]

        Dt = results.values[1,2]
        Tmax = results.values[-1,2]
        timedata = [Dt,Tmax]
    else:
        print("ERROR: INCORRECT EXPERIMENTAL FILE EXIT: in FUNCTION expextract")
        exit()

    return Evalues, ACsettings,scanrate,current,voltage,timedata

def paraextract(parameterestimates):
    paradic = {}
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase

    # this is for scientific notation
    lowercase = lowercase.replace("e", "")
    uppercase = uppercase.replace("E","")
    strings = lowercase+ uppercase
    for values in parameterestimates:
        x = values.split("=")
        y = letters2values(x[0])
        s = ""
        i = 0
        for letters in x[1]:
            if letters not in strings:
                s += letters
            else:
                break
            i += 1

        unit = ""
        for j in range(i,len(x[1])):
            unit += x[1][j]

        """Something to convert units into MECSim friendly units"""
        s = unitconversion(y,float(s),unit)
        print(y,s,unit)
        paradic.update({y:s})


    return paradic

def DNN_model_locloader(filename):
    # loads the file and sets stuff up
    dic = {}
    f = open(filename,"r")
    filelist = f.readlines()
    bayes_exist = False

    for lines in filelist:
        x = lines.split("=")
        label = x[0].strip(" \n\t")
        file = x[1].strip(" \n\t")
        # this is for the clustering
        if label != "bayesclusteringinfo":
            dic.update({label:file})
        else:
            bayesloc = file
            bayes_exist = True

    if not bayes_exist:
        bayesloc = None

    return dic,bayesloc

# loads the bayesian probability of the function
def clusterbayesloader(filename):
    f = open(filename)
    filelist = f.readlines()
    listofdic = []
    for lines in filelist:
        dic = {}
        x = lines.split("\t")
        for i in range(1,len(x)-1):
            y = x[i].strip(" \t\n")
            y = y.split(":")
            #if y[0] != "": # check to see if end has been reached
            dic.update({y[0]:float(y[1])})
        listofdic.append(dic)

    return listofdic