import string
from itertools import islice
from pandas import read_csv
import numpy as np
from scipy.fftpack import rfft, irfft, rfftfreq
from scipy.signal import hilbert as anal_hil_tran # NEED signal to make a smooth envolope
import matplotlib.pyplot as plt
import psycopg2
import os

# loads in a file of generic parameter values
def settingsload(filename):

    with open(filename) as f:
        file = f.readlines()

    numiter = int(file[0].split("=")[1])

    stdaccept = float(file[1].split("=")[1])

    return numiter, stdaccept

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

    reactmech = inputdic.get("-m")
    parameterestimates = inputdic.get("-p")

    return expfile,reactmech,parameterestimates

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

# defines automaticly the accuracy of the simulation
def simaccuracy(Tmax, ACmode,frequency):
    print("fix sim accuracy currently too lazy L381 greedy_mod.py")
    # dpoints = 2**Nsim
    if ACmode:
        if Tmax < 20:
            Nsim = 16
        else:
            Nsim = 17
    else:
        if Tmax < 20:
            Nsim = 14
        else:
            Nsim = 15
    return Nsim

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

def Reactparaallocator(reactmech):
    if reactmech == "E":
        allpara = [[11,1],[22,1],[33,1],[34,1],[35,1]]
        scaledparas = [[1.0,2,22,2]]
        functparas = []
        spaces = [34, 38, 47, 2, 1]  # These have been predetermined and ar of the form []
        Nreact = 1 # specifies the number of reaction lines
    elif reactmech == "EE":
        allpara = [[11,1],[22,1],[33,1],[34,1],[35,1],[33,2],[34,2],[35,2]]
        scaledparas = [[1.0, 2, 22,2],[1.0, 2, 22,3]]
        functparas = []
        spaces = [34, 38, 48, 3, 1]
        Nreact = 2
    elif reactmech == "EC":
        allpara = [[11, 1], [22, 1], [33, 1], [34, 1], [35, 1], [41, 2], [42, 2]]
        scaledparas = [[1.0, 2, 22, 2], [1.0, 2, 22, 3]]
        functparas = [[1],[1,6,1,7]]
        spaces = [34, 38, 48, 3, 1]
        Nreact = 2
    else:
        print("Incorrect labels selected please double check that this reaction mechanism is supported")

    return allpara,scaledparas,functparas, spaces, Nreact

def supivisedpriors(serverdata, ReactionMech,allparas):
    try:
        connection = psycopg2.connect(user=serverdata[0],
                                          password=serverdata[1],
                                          host=serverdata[2],
                                          port=serverdata[3],
                                          database=serverdata[4])

        cursor = connection.cursor()

        cursor.execute("""SELECT "inputfile" FROM "ExperimentalSettings" WHERE "ReactionMech" = %s """, (ReactionMech,))
        inputfileraw = cursor.fetchall()[0][0]  # This extracts the input for writterV_2 for MECSim

        inputfile = inputfileraw.split("\n") # seperates the lines

        i = 0
        for lines in inputfile:
            if lines.strip("\t ") == "varibles":
                st = i
            elif lines.strip("\t ") == "scalvar":
                end = i
                break
            i += 1
        varibles = inputfile[st+1:end]

        # remove empty lines JUST IN CASE
        hold = []
        j = 0
        while j != len(varibles):
            if varibles[j].strip("\t ") == "":
                hold.append(j)
            else:
                varibles[j] = varibles[j].strip(" ").split(",")
            j += 1

        for i in hold[::-1]: # needs to be on reversible order
            varibles.pop(i)

        priors = []

        for values in allparas:
            for stuff in varibles:
                if values[0] == int(stuff[3]) and  values[1] == int(stuff[4]): #checks same parameter
                    priors.append([float(stuff[0]),float(stuff[1]),int(stuff[2])]) # min, max log scale
                elif values[0] == int(stuff[3]):  # for the case where there isn't the stuff in the input file but its required
                    priors.append([float(stuff[0]), float(stuff[1]), int(stuff[2])])  # min, max log scale


    except (Exception, psycopg2.Error) as error:
        print("error,", error)
    finally:
        # This is needed for the
        if (connection):
            cursor.close()
            connection.close()

    # something here to double check that all para equals same length as priors and correct if not mainly for EE react mechs
    if len(priors) != len(allparas):
        print("priors")
        print(priors)
        print("parameters for reaction mechanism")
        print(allparas)
        print("Error: priors do not match up with all paras TERMINATING")
        exit()

    return priors

# gets the value of the pinned parameters
def varallocator(allpara,paraestdic,priors,scaledparas,functparas,E0values,spaces):

    En = len(E0values)
    pinnedpara = []

    # extracts and seperates the values in the input dictionary
    ii = 0
    feedpara = []
    for keys, items in paraestdic.items():
        if int(keys)< 10:
            pinnedpara.append([items,int(keys),1])
            ii += 1
        elif int(keys) == 21: #concentration
            pinnedpara.append([items, int(keys), 1])
            ii += 1
        else:
            feedpara.append(int(keys))

    var = []
    functionallist = []
    i = 0
    f = 1
    priorsdel = []
    for values in allpara:
        varin = []
        allocated = False
        if any(values[0] == j for j in feedpara):
            x = paraestdic.get(values[0])
            pinnedpara.append([x, int(values[0]), values[1]])

            # adjust priors
            if x <= priors[i][0] or x >= priors[i][1]:
                priorsdel.append(i)
                print("Parameter " + str(values[0])+", on reapeat line " +str(values[1]) + " pinned to given value as it is out of range")

            elif abs(x - priors[i][0]) < abs(priors[i][1] - x):
                priors[i][1] = 2*x - priors[i][0]
            else:
                priors[i][0] = 2*x - priors[i][1]

        else:
            if priors[i][2] == 0: # nonlog
                x = (priors[i][1] + priors[i][0])/2
            else:
                x = 10**((np.log10(priors[i][0]) + np.log10(priors[i][1]))/2)
            pinnedpara.append([x, int(values[0]), values[1]])

        # extracts the functional parameters
        if values[0] > 40 and values[0] < 50:
            functionallist.append(pinnedpara[-1])
            #changes the
            j = 1
            for stuff in functparas[1::]:
                if stuff[1] == i+1:
                    functparas[j][1] = f
                j += 1
        f += 1

        # sets up the var input
        varin.append(x)
        [varin.append(k) for k in priors[i]]
        [varin.append(k) for k in values]
        var.append(varin)

        # scaled parameters
        for j in scaledparas:
            if j[1] - 1 == i:
                pinnedpara.append([j[0]*pinnedpara[ii][0],j[2],j[3]])

        i += 1
        ii += 1

    # corrects the paired THESE functional parameters are a mess
    j = 0

    for stuff in functparas[1::]:

        if stuff[2] == 1:
            x1 = allpara[stuff[3]-1][0]
            x2 = allpara[stuff[3]-1][1]

            f = 1
            for the in var:
                if x1 == the[4] and x2 == the[5]:
                    functparas[j+1][3] = f
                f += 1
        j += 1

    # place E0 pinned in
    Neideal = 0
    i = 0
    estore = []
    for values in pinnedpara:
        if values[1] == 33:
            estore.append(i)
            Neideal += 1
        i += 1

    vestore = []
    for ll in range(len(var)):
        if var[ll][4] == 33:
            vestore.append(ll)

    if Neideal == En: # case where all possible E0 have been found
        i = 0
        for values in estore:
            pinnedpara[values][0] = E0values[i]
            i += 1

        i = 0
        for values in vestore:
            var[values][0] = E0values[i]
            var[values][1] = E0values[i] - 0.03
            var[values][2] = E0values[i] + 0.03
            i += 1

    elif Neideal > En and En != 0: # case where there is less then ideal cases (Only EE case currently)
        i = 0
        for values in estore:
            pinnedpara[values][0] = E0values[0]
            i += 1

        i = 0
        for values in vestore:
            var[values][0] = E0values[0]
            var[values][1] = E0values[0] - 0.03
            var[values][2] = E0values[0] + 0.03
            i += 1

    else:
        print("Error no formal potentials found in experimental data")

    #Functionalparameters
    #print("Fix functional parameters as required for kb and kf pinning")
    if len(functionallist) != 0:
        l  = [0 for i in range(len(allpara))]
        j = 0
        for i in range(len(allpara)):
            # extracts the functional parameters
            if allpara[i][0] > 40 and allpara[i][0] < 50:
                l[i] = functionallist[j][0]
                j += 1

        funclist = listin_func(l, functparas, functionallist)

        for values in funclist:
            for i in range(len(pinnedpara)):
                if pinnedpara[i][1] == 41 and pinnedpara[i][2] == values[-1]: # checks for value and right row
                    pinnedpara[i][0] = values[0]
                    pinnedpara[i][1] = 31   # sets to kf
                elif pinnedpara[i][1] == 42 and pinnedpara[i][2] == values[-1]: # checks for value and right row
                    pinnedpara[i][0] = values[1]
                    pinnedpara[i][1] = 32   # sets to kb

    # del priors and var and varibles outside the range
    #[var.pop(i) for i in priorsdel] this can be done by anotherfunction
    [priors.pop(i) for i in priorsdel]

    # changes value and repeat into a location in data Master.inp
    pinnedpara = greedymodline_assigner(pinnedpara, spaces)

    # changes value and repeat into a location in data Master.inp does it for var as well
    var = greedymodline_assigner(var, spaces)

    # temperature correction (assumes 21 C )
    if paraestdic.get(1) == None:
        pinnedpara.append([21 + 273.15,1,0]) # assume temp is 21C unless otherwise stated

    return pinnedpara, var, priorsdel

# does the functional code to an input list
def listin_func(val_in, funcvar, funcvar_holder): #(val_in, funcvar, funcvar_holder,data,var)
    funclist = []
    i = 0
    while i != funcvar[0][0]:
        Nfunc = int(funcvar[i+1][1]) - 1  # gets varible row
        if funcvar[i + 1][2] == 0:  # if functional parameter is unpaired
            if funcvar_holder[i][1] == 41:  # R functional parameter
                # sets Kf
                Kf = val_in[Nfunc] * np.sin(funcvar[i + 1][3] * (np.pi / 180))
                # sets kb
                Kb = val_in[Nfunc] * np.cos(funcvar[i + 1][3] * (np.pi / 180))

                funclist.append(np.array([Kf, Kb, 1, funcvar_holder[i][2]]))

            elif funcvar_holder[i][1] == 42:  # theta functional parameter

                Kf = funcvar[i + 1][3] * np.sin(val_in[Nfunc] * (np.pi / 180))
                # sets kb
                Kb = funcvar[i + 1][3] * np.cos(val_in[Nfunc] * (np.pi / 180))

                funclist.append(np.array([Kf, Kb, 1, funcvar_holder[i][2]]))

            else:
                pass

        else:  # if functional parameter is paired
            Ncoup = int(funcvar[i + 1][3]) - 1  # gets paired parameter listin input
            if funcvar_holder[i][1] == 41:  # R functional parameter

                Kf = val_in[Nfunc] * np.sin(val_in[Ncoup] * (np.pi / 180))
                # sets kb
                Kb = val_in[Nfunc] * np.cos(val_in[Ncoup] * (np.pi / 180))

                funclist.append(np.array([Kf, Kb, 1, funcvar_holder[i][2]]))

            elif funcvar_holder[i][1] == 42:  # theta functional parameter

                Kf = val_in[Ncoup] * np.sin(val_in[Nfunc] * (np.pi / 180))
                # sets kb
                Kb = val_in[Ncoup] * np.cos(val_in[Nfunc] * (np.pi / 180))

                funclist.append(np.array([Kf, Kb, 1, funcvar_holder[i][2]]))

            """elif funcvar_holder[i][4] == 43:  # E as a function of two other varibles

                funcval = val_in[Nfunc]

                Ncoup1 = int(funcvar[i + 1][3]) - 1
                Ncoup2 = int(funcvar[i + 1][4]) - 1

                Ncode1 = int(var.iloc[Ncoup1][4])
                Ncode2 = int(var.iloc[Ncoup2][4])

                if Ncode1 == 33:
                    E1 = val_in[Ncoup1]
                    E2 = val_in[Ncoup2]
                else:
                    print("ERROR: please put the formal potential in second position")

                if Ncode2 == 3: #case where Erev is varible
                    Econc = float(data.iloc[2][0])
                    if Econc > E1: # potential going negitive
                        E0 = E1 - funcval*abs(E1-E2)
                    else:
                        E0 = E1 + funcval * abs(E1 - E2)

                elif Ncode2 == 2: #case where Estart is varible
                    Econc = float(data.iloc[3][0])
                    if Econc > E1: # potential going positive
                        E0 = E1 + funcval*abs(E1-E2)
                    else:
                        E0 = E1 - funcval * abs(E1 - E2)
                E2 = val_in[Ncoup2]

                funclist.append(np.array([E0, 0.0, 2, funcvar_holder[i][5]])) #second input is a place holder

            elif funcvar_holder[i][4] == 44: # scalar function
                funcval = val_in[Nfunc]
                Ncoup = int(funcvar[i + 1][3]) - 1
                val = val_in[Ncoup]*funcval
                number = int(var.iloc[Ncoup][5]) + int(var.iloc[Nfunc][5]) - 1 #position that value goes in at
                funclist.append(np.array([val,0.0, 3, number])) #second input is a place holder

            else:
                pass"""

        # hunt for dublicates
        if funcvar[i + 1][2] == 1:  # checks if current varible is coupled
            k = i + 1  # makes sure it doesn't delete itself
            while k != funcvar[0][0] + 1:
                if funcvar[i + 1][3] == funcvar[k][1]:
                    del funcvar[k]  # deletes dublicate
                    funcvar[0][0] -= 1  # fills deleted input

                else:
                    k += 1

        else:
            pass

        i += 1

    return funclist

# Assigns the varibles to there respective lines in the MECSim settings
def greedymodline_assigner(pinnedpara,  spaces):
    # sets the value for var
    i = 0
    while i != len(pinnedpara):

        # Misc varibles
        if pinnedpara[i][-2] == 11:  # resistance
            pinnedpara[i][-1] = 1

        elif pinnedpara[i][-2] == 1: # temp
            pinnedpara[i][-1] = 0

        elif pinnedpara[i][-2] == 2:  # Estart
            pinnedpara[i][-1] = 2
        elif pinnedpara[i][-2] == 3: # Erev
            pinnedpara[i][-1] = 3
        elif pinnedpara[i][-2] == 4: # scanrate
            pinnedpara[i][-1] = 5
        elif pinnedpara[i][-2] == 6: # Surface Area
            pinnedpara[i][-1] = spaces[0] - 9
        elif pinnedpara[i][-2] == 7 or  pinnedpara[i][-2] == 8 : # amp and frequency
            pinnedpara[i][-1] = spaces[0] + pinnedpara[i][-1] +1
        elif pinnedpara[i][-2] == 12:  # kinematic vis
            pinnedpara[i][-1] = spaces[0]

        elif pinnedpara[i][-2] == 13:  # NOISE PARAMETER
            pass

        elif pinnedpara[i][-2] > 20 and pinnedpara[i][-2] < 30:  # Solution block propities
            pinnedpara[i][-1] = spaces[1] + pinnedpara[i][-1] - 1

        elif pinnedpara[i][-2] > 30 and pinnedpara[i][-2] < 40:  # Kinetic propities
            pinnedpara[i][-1] = spaces[2] + pinnedpara[i][-1] - 1

        elif pinnedpara[i][-2] == 41 or pinnedpara[i][-2] == 42:  # functional propities for Keq (41= r, 42 = theta)
            pinnedpara[i][-1] = spaces[2] + pinnedpara[i][-1] - 1

        elif pinnedpara[i][-2] == 43:  # E0 when the range is defined by two other varibles
            pinnedpara[i][-1] = spaces[2] + pinnedpara[i][-1] - 1
        elif pinnedpara[i][-2] == 44:  # varing a function as a scalar of another varible allot this at a seperate section
            pass

        elif pinnedpara[i][-2] > 50 and pinnedpara[i][-2] < 60:  # capacitance propities for Keq (41= r, 42 = theta)
            pinnedpara[i][-1] = spaces[2] - 6 + pinnedpara[i][-1] - 50
            # 51 = C0, 52 = C1, 53 = C2, 54 = C3, 55 = C4, 56 = Scalar*(C)


        else:
            print('Error in script_generator function line_assigner')

        i += 1

    return pinnedpara


# Automaticly identifies if AC or DC methods should be used then allocates it
# OLD OLD CODE
def ACDC_method(AC_amp, op_settings):
    S = np.sum(AC_amp)
    DCAC_method = [0, 0, 0, 0]  # here to assign the fitting method and ACDC method

    # Optimization settings
    # op_settings = [0,0]     #prealocation
    DCAC_method[1] = op_settings[0]  # sets the fitting function with {0 = absfit, 1 = %fit}
    DCAC_method[2] = op_settings[1]  # Sets the number of cores
    DCAC_method[3] = op_settings[2]  # Sets the decimation number

    if S != 0:
        DCAC_method[0] = 1

    else:

        DCAC_method[0] = 0

    return DCAC_method

def functionholder(var):

    funcvar_holder = []
    # issolates up the functional parameters
    i = 0
    while i != len(var.iloc[:][0]):
        if var.iloc[i][4] == 41 or var.iloc[i][4] == 42:
            x = var.iloc[i].values
            x = np.append(x, i + 1)  # adds the varible column data to the area we need
            funcvar_holder.append(x)

        else:
            pass
        i += 1

    return funcvar_holder

# Assigns the varibles to there respective lines in the MECSim settings
def scaledline_assigner( spaces, scalvar):

    # sets the values for scalvar
    i = 1
    while i != len(scalvar):

        # Misc varibles
        if scalvar[i][2] == 11:  # resistance
            scalvar[i][3] = 1

        elif scalvar[i][2] == 12:  # kinematic vis
            scalvar[i][3] = spaces[0]

        elif scalvar[i][2] == 13:  # NOISE PARAMETER
            pass

        elif scalvar[i][2] > 20 and scalvar[i][2] < 30:  # Solution block propities
            scalvar[i][3] = spaces[1] + scalvar[i][3] - 1

        elif scalvar[i][2] > 30 and scalvar[i][2] < 40:  # Kinetic propities
            scalvar[i][3] = spaces[2] + scalvar[i][3] - 1

        else:
            print('lol what you do now')

        i += 1

    return scalvar

# truncates and passes around experimental harmonics and find truncation points
def EXPharmtreatmentOG(EX_hil_store,sigma,Extime, truntime, op_settings):

    nEX = EX_hil_store.shape[1]

    E1 = find_nearest(np.linspace(0, Extime, nEX), truntime[0])
    Ndeci = op_settings[2]

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
    EX_hil_store = EX_hil_store[:, E1:E2]
    # need to identify correct points for simulation and experimental so that its the same time spot then pass identified decimated simulation values to MEC_set

    # Truncates the sigma file if applicable to method
    if isinstance(sigma, bool) or sigma == None:  # sigma array exists  """ or method == 'Baye_HarmPerFit'"""
        pass
    else:
        sigma = sigma[:, E1:E2]
        # Somefunction that truncates the sigma values at set time points

    return EX_hil_store, sigma, Nsimdeci, Nex

# function for checking if the output file
def outputfilegenertor(outputfname):

    file = True
    i = 0
    while file:
        try:
            if i == 0:
                filename = outputfname
                os.makedirs(filename)
            else:
                filename = outputfname +"_V" +str(i)
                os.makedirs(filename)
            file = False
        except:
            print("file already exists")
        finally:
            i += 1

    return filename


