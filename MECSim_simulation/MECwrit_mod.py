from scipy.fftpack import rfft, irfft, rfftfreq
from scipy.signal import hilbert as anal_hil_tran # NEED signal to make a smooth envolope
import numpy as np
import psycopg2
from Script_generator import *
import time
import datetime
import ML_signal_processing as ML_sp
import matplotlib.pyplot as plt
import NN_posttraining_mod as NN_pt
import os
import window_test as Wint

import mecsim  # imports modulated mecsim

def inputloader(input_name):

    with open(input_name) as f:
        lines = f.readlines()

    #serverdata in format ["serverdata"]

    serverdata = inputstrip(lines[0])
    serverdata = serverdata.split(",")
    x = []
    # order is [user, password, host, port, database]
    for values in serverdata:
        x.append(values.strip(" "))
    serverdata = x

    inputMEC = inputstrip(lines[1])

    ReactionMech = inputstrip(lines[2])

    RedoxOrder = inputstrip(lines[3])

    x = inputstrip(lines[4])
    AC_case = [x=="True" or x=="true"]
    AC_case = AC_case[0]

    AutoNharm = float(inputstrip(lines[5]))

    Cnoise = float(inputstrip(lines[6]))

    N = int(inputstrip(lines[7]))

    seperating = inputstrip(lines[8]).strip("[]").split(",")
    seperating = [int(seperating[0]),int(seperating[1])]

    decilist = inputstrip(lines[9]).strip("[]").split(",")
    decilist = [int(decilist[0]), int(decilist[1])]

    samplertype = inputstrip(lines[10])

    windowing = inputstrip(lines[11])

    guass_std = float(inputstrip(lines[12]))

    bandwidth = inputstrip(lines[13]).split(",")

    holder = []
    list = []
    for values in bandwidth[0:int(len(bandwidth)/2)]:
        x = values.strip("[] ")
        list.append(float(x))
    holder.append(list)
    list = []
    for values in bandwidth[int(len(bandwidth) / 2):]:
        x = values.strip("[] ")
        list.append(float(x))
    holder.append(list)
    bandwidth = np.array(holder)

    i = 14
    for line in lines[i:]:
        if line.strip("\n\t ") == "varibles":
            jj = i
            break
        i += 1

    x = inputstrip(inputstrip(lines[14]))
    nondim = [x == "True" or x == "true"]
    nondim = nondim[0]
    print(nondim)

    trun = float(inputstrip(lines[15]))

    varibles = []
    j = jj +1
    for line in lines[j:]:
        if line != "\n":
            varibles.append(funcstrip(line))
        else:
            break
        jj += 1

    i = jj
    for line in lines[i:]:
        if line.strip("\n\t ") == "scalvar":
            jj = i
            break
        i += 1

    scalvar = []
    j = jj +1
    #scalvar.append(int(lines[j].strip("\n\t ")))
    jj = j
    for line in lines[j:]:
        if line != "\n":
            scalvar.append(funcstrip(line))
        else:
            break
        jj += 1

    i = j
    for line in lines[j:]:
        if line.strip("\n\t ") == "funcvar":
            jj = i
            break
        i += 1

    funcvar = []
    j = jj + 1
    #funcvar.append(int(lines[j].strip("\n\t ")))
    for line in lines[j:]:
        if line != "\n":
            funcvar.append(funcstrip(line))
        else:
            break
        j += 1

    return serverdata, inputMEC, ReactionMech, RedoxOrder,AC_case , AutoNharm, Cnoise, N, seperating, decilist, \
           samplertype, windowing, guass_std, bandwidth, varibles, scalvar, funcvar, nondim, trun

def funcstrip(x):
    y = x.strip("\n\t ")
    y = y.split(",")
    y1 = []
    for values in y:
        x2 = values.strip("\n\t ")
        y1.append(float(x2))

    return y1


def inputstrip(x):
    y = x.strip("\n\t ")
    y = y.split("#")[0]
    y = y.strip("\n\t ")
    y = y.split("=")[-1]
    y = y.strip(" \t")
    return y


def scaler(listin,lowerbound,upperbound,meanbound,logstyle):

    Nvar = len(listin)

    for j in range(Nvar):
        if logstyle[j] == 0:  # Non-logaritmic scalable constant

            listin[j] = (upperbound[j] - lowerbound[j])/2 * listin[j] + meanbound[j]  # Parameter scaler

        elif logstyle[j] == 1:  # logaritmic scalable constant

            listin[j] = 10 ** ((np.log10(upperbound[j] / lowerbound[j])/2) * listin[j] + np.log10(
                        meanbound[j]))  # Parameter scaler

        else:
            print('please put a 0 or 1 in the right input')

    return listin

class MECSim_SQL():
    # preallocation of self
    kwargs = {}

    def __init__(self,*args, **kwargs):

        # args = [Excurr, Method]

        #kwargs is the MEC_SET
        self.kwargs = kwargs
        self.AC_freq = kwargs.get("AC_freq")
        self.bandwidth = kwargs.get("bandwidth")
        self.spaces = kwargs.get("spaces")
        self.deltatime = kwargs.get('deltatime') # gets the time intervals for simulation
        self.nsimdeci = kwargs.get('nsimdeci')
        self.decilist = kwargs.get('decilist')
        self.EXPsetrow  = kwargs.get("sqlExpinput")
        self.serverdata = kwargs.get("serverdata")
        self.AutoNharm = kwargs.get("AutoNharm")
        self.filter_hold = kwargs.get("filter_hold")
        self.trunvec = kwargs.get("trunvec")    # this has been changed to a percentage value
        self.AC_case = kwargs.get("AC_case")
        self.windowing = kwargs.get("windowing")
        self.guass_std = kwargs.get("guass_std")
        # surface confined object
        data = kwargs.get("data")
        x = []
        for i in range(self.spaces[3]):
            y = data.iloc[int(self.spaces[1] + i)][0].split(",")
            x.append(int(y[2]))
        self.surfconf = x
        self.nondim = kwargs.get("nondim")
        # defines the reaction mechanism type
        x = []
        i = 0

        while i + self.spaces[2] != len(data):
            y = data.iloc[int(self.spaces[2] + i)][0].split(",")
            x.append(int(y[0]))
            i += 1
        self.typelist = x

        # checks for frequency as a varible
        self.freqvar = kwargs.get("freqvar")
        if self.freqvar[0]:
            self.bandwidth = self.bandwidthchanger(self.bandwidth)

    # required to change the broadening of peaks due to scanrate
    def bandwidthchanger(self,bandwidth):
        bandwidth2 = [bandwidth]
        bandwidth2.append(bandwidth*1.5) #12
        bandwidth2.append(bandwidth*2)  #18
        bandwidth2.append(bandwidth*3)  #18
        bandwidth2.append(bandwidth * 3)   #60
        bandwidth2.append(bandwidth * 3)   #72

        return bandwidth2

    # these are all required such that frequency and scanrate couple and analytical is to difficult. multiple DNNs
    # Will be reuired for higher faster DC regions
    def bandwidthallocator(self,freq):

        if freq <= 12:
            bandwidth = self.bandwidth[0]
        elif freq > 12 and freq <= 24:
            bandwidth = self.bandwidth[1]
        elif freq > 24 and freq <= 34:
            bandwidth = self.bandwidth[2]
        elif freq > 34 and freq <= 45:
            bandwidth = self.bandwidth[3]
        elif freq > 45 and freq <= 75:
            bandwidth = self.bandwidth[4]
        elif freq >= 75:
            bandwidth = self.bandwidth[5]
        else:
            print("incorrect parameters")
            print(freq)

        return bandwidth

    def sql_loopr(self, *args, **kwargs):
        t1 = time.time()
        # The below is a really quick fix for the paraallisaion o pass reaction ID hrough pool but im lazy
        Reaction_ID = args[0][-1] # pops of the reaction ID then leaves he passed args
        args = args[0][0:-1]
        Scurr, hil_store, data = self.simulate(*[args])

        # SQl stuff current work in progress
        sqlharm, Nharm = self.sql_harmonics(Reaction_ID, Scurr, hil_store)
        sqlparadic = self.sql_paras(Reaction_ID, self.EXPsetrow, data, self.spaces,Nharm)

        self.SQL_updater(sqlparadic,sqlharm)
        print(time.time() - t1)

        return

    def simulate(self, *args):
        # Run a simulation with the given parameters for the
        # given times
        # and return the simulated value

        #wll need to print the MECSim nout o double check its going in and out at the same rate

        Scurr, data = ML_sp.Iterative_MECSim_Curr(args[0], **self.kwargs)

        #print("put something here for nondimensionalisation L271 MECwrittermod")
        if self.nondim:
            if any(self.surfconf) == 1:
                temp = float(data.iloc[0][0])
                scanrate = float(data.iloc[5][0])
                surfaceconc = data.iloc[int(self.spaces[1])][0].split(',') # breaks array input into sections
                surfaceconc =float(surfaceconc[0])
                area = float(data.iloc[self.spaces[0] - 9][0])
                currentnondimsurfconfined(Scurr,area, surfaceconc, scanrate, temp)
            else:
                """MIGHT NOT BE RIGHT FOR THE SURCFACE CONFINDED CASE"""
                area = float(data.iloc[self.spaces[0] - 9][0])
                diff = float(data.iloc[self.spaces[1]][0].split(",")[1])
                conc = float(data.iloc[self.spaces[1]][0].split(",")[0])
                Scurr,constant = currentnondim(Scurr,area, diff, conc)
                # remember the nondim decreses the powspectrum uniformly in the so you need to subtract the below for it to be right
                #constant = np.log10(constant)
        else:
            constant = 0

        if self.AC_case:
            if self.freqvar[0]: #if freq var True
                bandwidth = self.bandwidthallocator(args[0][self.freqvar[1]])

                #print("wee")
                #print(args[0][self.freqvar[1]])
                #print(data)
                Nsimlength = len(Scurr)
                Dt = outputsetterDT(data)/Nsimlength # needs this as a bunch of stuf changes
                #print(Dt)
                # 10 is the max number of harmonics
                nsimdeci,frequens = Simerharmtunc(Nsimlength, Dt, bandwidth, 10, [args[0][self.freqvar[1]]])
                #fft_res is returned for efficentcy
                Nharm, fft_res = harmoniccounter(Scurr,nsimdeci,self.AutoNharm)

                """print(args[0][self.freqvar[1]])
                plt.figure()
                plt.plot(frequens[0:5000], fft_res[0:5000])
                plt.savefig("figure2.png")
                plt.close()
                exit()"""

                if Nharm != 0:
                    bandwidth = bandwidth[:][:Nharm+2]
                else:
                    bandwidth = bandwidth[:][:1]

                #Required asfreq varies
                filter_hold = windowfunctioning(self.windowing, frequens, bandwidth, self.guass_std, args[0][self.freqvar[1]], Nharm)
                #as number of harmonics is gathered from the bandwidth need to edit another bndwidth to pss to calcs
                hil_store = Wint.windowed_harm_gen(fft_res, bandwidth, self.spaces[4], filter_hold)
            else:
                # fft_res is returned for efficentcy
                Nharm, fft_res = harmoniccounter(Scurr, self.nsimdeci, self.AutoNharm)
                if Nharm != 0:
                    bandwidth = self.bandwidth[:, 0:Nharm+1]
                else:
                    bandwidth = self.bandwidth[:, 0:1]
                # as number of harmonics is gathered from the bandwidth need to edit another bndwidth to pss to calc
                hil_store = Wint.windowed_harm_gen(fft_res, bandwidth, self.spaces[4], self.filter_hold)

            # trun = 0.4 # end point truncation time
            Exptime = Dt * Nsimlength  # calcs the overall exp time
            truntime = Exptime * self.trunvec  # percentage based truncation
            Timesim = np.linspace(0, Exptime, num=Nsimlength)

            int_s = Wint.find_nearest(Timesim, truntime)
            int_e = Wint.find_nearest(Timesim, Exptime - truntime)
            trunvec = [int(int_s), int(int_e)]

            if Nharm != 0:

                for i in [0, 1]:
                    hil_store[i][:] = Wint.Average_noiseflattern(hil_store[i][:],trunvec,20)

                if Nharm != 1:
                    Ndc = 2  # here to exclude DC and fundimenal
                    # truncates to zero
                    for i in range(Ndc,Nharm+2):
                        hil_store[i][:trunvec[0]] = 0
                        hil_store[i][trunvec[1]:] = 0

            else:
                hil_store[0][:] = Wint.Average_noiseflattern(hil_store[0][:], trunvec,20)

        else: # DC case
            hil_store = None

        return Scurr, hil_store, data


    def SQL_updater(self,sqlparadic,sqlharm):

        serverdata = self.serverdata

        try:

            connection = psycopg2.connect(user=serverdata[0],
                                          password=serverdata[1],
                                          host=serverdata[2],
                                          port=serverdata[3],
                                          database=serverdata[4])

            cursor = connection.cursor()

            # put in the parameers where they sould go
            qinput = """UPDATE "Simulatedparameters" SET "ReactionMech" = %(ReactionMech)s,  "EXPsetrow" = %(EXPsetrow)s,  "Temp" = %(Temp)s, 
                          "Ru" = %(Ru)s,  "Electrontransmech" = %(Electrontransmech)s,  "CapDL" = %(CapDL)s,  "Conc" = %(Conc)s,  "Diff" = %(Diff)s,
                          "surfconfined" = %(surfconfined)s,  "kbackward" = %(kbackward)s,  "kforward" = %(kforward)s,  "formalE" = %(formalE)s,
                          "ksreal" = %(ksreal)s, "alpha" = %(alpha)s, "type" = %(type)s, 
                           "RedoxOrder" = %(RedoxOrder)s, "Reactionmatrix" = %(Reactionmatrix)s, "NumHarm" = %(NumHarm)s,
                           "Estart" = %(Estart)s, "Eend" = %(Eend)s,"scanrate" = %(scanrate)s,"Exptime" = %(Exptime)s, 
                           "ElectrodeArea" = %(ElectrodeArea)s,"sineamp" = %(sineamp)s,"sinefreq" = %(sinefreq)s WHERE "Reaction_ID" = %(Reaction_ID)s"""
            cursor.execute(qinput, sqlparadic)

            # put in the harmonic informaion
            qinput = """UPDATE "HarmTab" SET "TotalCurrent" = %(TotalCurrent)s, "NumHarm" = %(NumHarm)s,
                    "HarmCol0" = %(HarmCol0)s, "HarmCol1" = %(HarmCol1)s, "HarmCol2" = %(HarmCol2)s, 
                    "HarmCol3" = %(HarmCol3)s, "HarmCol4" = %(HarmCol4)s, "HarmCol5" = %(HarmCol5)s, "HarmCol6" = %(HarmCol6)s,
                     "HarmCol7" = %(HarmCol7)s, "HarmCol8" = %(HarmCol8)s,
                      "Reaction_ID" = %(Reaction_ID)s, "EXPsetrow" = %(EXPsetrow)s WHERE "Reaction_ID" = %(Reaction_ID)s"""
            cursor.execute(qinput, sqlharm)

            connection.commit()  # the commit is needed to save changes

        except (Exception, psycopg2.Error) as error:
            print("error message from sql_updater,", error)
        finally:
            # This is needed for the
            if (connection):
                cursor.close()
                connection.close()

        return

    # something to
    def sql_harmonics(self, Reaction_ID,Scurr,hil_store):
        # self.decilist decimation for storage to sql server [2**Currdeci, 2**harmonic decimation]
        sqlharm = {}

        decicurr = int(len(Scurr) / (2 ** self.decilist[0]))

        if self.AC_case == True:
            Nharm = len(hil_store) - 1 # number of harmonics excluding dc component
            deciharm = int(len(hil_store[0]) / (2 ** self.decilist[1]))

        else:
            Harmpickle = None
            Nharm = None

        sqlharm.update({"EXPsetrow": self.EXPsetrow})
        sqlharm.update({"NumHarm":Nharm})

        sqlharm.update({"Reaction_ID": Reaction_ID})
        sqlharm.update({"TotalCurrent":Scurr[::decicurr].tolist()})

        if self.AC_case == True:

            if Nharm < 13:
                for i in range(Nharm + 1):
                    l = "HarmCol" + str(i)
                    sqlharm.update({l: hil_store[i][::deciharm].tolist()})

                # add nonetype enteries for the rest of the harmonics
                for i in range(Nharm + 1,13):
                    l = "HarmCol" + str(i)
                    sqlharm.update({l: None})
            else:
                print("WARNING: more Harmonics then 12 where noted")
                for i in range(13):
                    l = "HarmCol" + str(i)
                    sqlharm.update({l: hil_store[i][::deciharm].tolist()})
        else: # DC Case
            for i in range(13):
                l = "HarmCol" + str(i)
                sqlharm.update({l: None})

        return sqlharm, Nharm

    # Something to sort out the parameters to put into the sql database
    def sql_paras(self, Reaction_ID, EXPsetrow, data, spaces,Nharm):

        sqlparadic = {}

        # Number of harmonics present per simulation
        sqlparadic.update({"NumHarm": Nharm})

        # Reaction Mechanism
        sqlparadic.update({"ReactionMech":self.kwargs.get("ReactionMech")})
        # Reaction ID
        sqlparadic.update({"Reaction_ID" : Reaction_ID})
        # Row in Expermental settings where all the data is stored
        sqlparadic.update({"EXPsetrow": EXPsetrow})
        # Temp
        sqlparadic.update({"Temp": float(data.iloc[0])})
        # Ru
        sqlparadic.update({"Ru": float(data.iloc[1])})
        # The electron transfer mechanism (0 = Butler-Volmer; 1 = Marcus Theory)
        sqlparadic.update({"Electrontransmech": int(data.iloc[9])})
        # "CapDL"
        allocator = ["CapDL", False]
        x = sqldatacolector(data, spaces, allocator, self.surfconf, self.typelist)
        sqlparadic.update({"CapDL": x})
        # RedoxOrder
        sqlparadic.update({"RedoxOrder": self.kwargs.get("RedoxOrder")})
        # "surfconfined"
        sqlparadic.update({"surfconfined": self.surfconf})
        # reaction type
        sqlparadic.update({"type": self.typelist})

        """ Some parameter function for solution properties """

        # "Conc"
        allocator = ["sol","conc"]
        x = sqldatacolector(data,spaces,allocator,self.surfconf,self.typelist)
        sqlparadic.update({"Conc": x})

        # "Diff"
        allocator = ["sol", "diff"]
        x = sqldatacolector(data,spaces,allocator,self.surfconf,self.typelist)
        sqlparadic.update({"Diff": x})

        """ Some parameter function for ET identificaion """

        # "kbackward"
        """ "kbackward" set i belevie it needs to real[] array so  something """
        allocator = ["kinetics", "kb"]
        x = sqldatacolector(data,spaces,allocator,self.surfconf,self.typelist)
        sqlparadic.update({"kbackward": x})
        # "kforward"
        """ "kforward" set i belevie it needs to real[] array so  something """
        allocator = ["kinetics", "kf"]
        x = sqldatacolector(data,spaces,allocator,self.surfconf,self.typelist)
        sqlparadic.update({"kforward": x})
        # "formalE"
        """ "formalE" set i belevie it needs to real[] array so  something """
        allocator = ["kinetics", "e0"]
        x = sqldatacolector(data,spaces,allocator,self.surfconf,self.typelist)
        sqlparadic.update({"formalE":x})
        # "ksreal"
        """ "ksreal" set i belevie it needs to real[] array so  something """
        allocator = ["kinetics", "k0"]
        x = sqldatacolector(data,spaces,allocator,self.surfconf,self.typelist)
        sqlparadic.update({"ksreal": x})
        # "alpha"
        allocator = ["kinetics", "alp"]
        x = sqldatacolector(data,spaces,allocator,self.surfconf,self.typelist)
        sqlparadic.update({"alpha": x})

        # "Reactionmatrix" matrix of the reaction occurring saved as a array of strings
        # (that will need to be filtered and standardized) with each row being a new entry
        # charcer varying[]
        allocator = ["Reactmech", False]
        x = sqldatacolector(data,spaces,allocator,self.surfconf,self.typelist)
        sqlparadic.update({"Reactionmatrix": x})

        sqlparadic.update({"Estart": float(data.iloc[2][0])})
        sqlparadic.update({"Eend": float(data.iloc[3][0])})
        sqlparadic.update({"scanrate": float(data.iloc[5][0])})
        timer = outputsetterDT(data)
        sqlparadic.update({"Exptime": float(timer)})

        sqlparadic.update({"ElectrodeArea": float(data.iloc[spaces[0] - 9][0])})

        x = data.loc[spaces[0] +2][0]
        x = x.split(",")
        sinamp = float(x[0])
        sinfreq = float(x[1])

        sqlparadic.update({"sineamp": sinamp})
        sqlparadic.update({"sinefreq": sinfreq})


        return sqlparadic

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


#posotion allocator for the sql sorter allocator hs the hsape of [numergroup,which para left to right]
def sqldatacolector(data,spaces,allocator,surfconf,typelist):
    # all inputs should be set up for sql ready format
    if allocator[0] == "sol":
        """ "Conc" set i belevie it needs to real[] array so something"""
        if allocator[1] == "conc":
            x = []
            for i in range(spaces[3]):
                y = data.iloc[int(spaces[1] + i)][0].split(",")
                x.append(float(y[0]))

        elif allocator[1] == "diff":    #""" "Diff" set i belevie it needs to real[] array so something"""
            x = []
            i = 0
            for sol in surfconf:
                if sol == 0:
                    y = data.iloc[int(spaces[1] + i)][0].split(",")
                    x.append(float(y[1]))
                else:
                    x.append(None)
                i += 1
        else:
            print("Error in sql allocator for solution")


    elif allocator[0] == "kinetics":

        """for these i need something to check if each row parameters is used and pass a mull input which in python is just None"""
        Nk = len(typelist)
        if allocator[1] == "kb":
            x = []
            for i in range(Nk):
                if typelist[i] == 0:
                    x.append(None)
                else:
                    y = data.iloc[int(spaces[2] + i)][0].split(",")
                    x.append(float(y[spaces[3]+2]))

        elif allocator[1] == "kf":
            x = []
            for i in range(Nk):
                if typelist[i] == 0:
                    x.append(None)
                else:
                    y = data.iloc[int(spaces[2] + i)][0].split(",")
                    x.append(float(y[spaces[3]+1]))

        elif allocator[1] == "e0":
            x = []
            for i in range(Nk):
                if typelist[i] != 0:
                    x.append(None)
                else:
                    y = data.iloc[int(spaces[2] + i)][0].split(",")
                    x.append(float(y[spaces[3]+3]))

        elif allocator[1] == "k0":
            x = []
            for i in range(Nk):
                if typelist[i] != 0:
                    x.append(None)
                else:
                    y = data.iloc[int(spaces[2] + i)][0].split(",")
                    x.append(float(y[spaces[3]+4]))

        elif allocator[1] == "alp":
            x = []
            for i in range(Nk):
                if typelist[i] != 0:
                    x.append(None)
                else:
                    y = data.iloc[int(spaces[2] + i)][0].split(",")
                    x.append(float(y[spaces[3]+5]))

        else:
            print("Error in sql allocator for kinetics")

    elif allocator[0] == "CapDL":
        x = []
        for i in range(5):
            y = data.iloc[int(spaces[2] + i-5)][0]
            x.append(float(y))

    elif allocator[0] == "Reactmech":
        x = []
        for i in range(len(typelist)):
            s = ""
            y = data.iloc[int(spaces[2] + i)][0]
            y = y.replace(" ","")
            y = y.replace("\t", "")
            y = y.split(",")
            for i in range(spaces[3]): # goes through all the solution values
                if i == 0:
                    s += y[1+i]
                else:
                    s += "," + y[1+i]
            x.append(s)

    else:
        print("Error in sql allocator 0th input")

    return x

# just a quick thing to get data ino a format we can use
def datasetter(*args,**kwargs):

    val_in = args[0]  # extracts args to the value being modified as a list

    # Extracts the varibles
    data = kwargs.get('data')
    var = kwargs.get('var')
    spaces = kwargs.get('spaces')
    # Exp_data = kwargs.get('Exp_data')
    scalvar = kwargs.get('scalvar')
    DCAC_method = kwargs.get('DCAC_method')
    funcvar = kwargs.get('funcvar')
    funcvar_holder = kwargs.get('funcvar_holder')
    cap_series = kwargs.get('cap_series')
    # extracts and removes the counter
    # counter = val_in[-1]
    # val_in = val_in[0:-1]
    # Ncore = DCAC_method[2]

    data = Var2data(data, var, val_in, spaces)  # turns the varibles into data

    # sets up for scalar dependant varibles
    if scalvar[0][0] != 0:
        data = Scalvar2data(scalvar, val_in, data, spaces)
    else:
        pass

    if funcvar[0][0] != 0:
        funclist = ML_sp.listin_func(val_in, funcvar, funcvar_holder)
        data = funcvar2data(funclist, spaces, data)
    else:
        pass

    # checks to see if capacitance is present

    if cap_series[0]:

        data = cap2data(val_in, var, spaces, data, cap_series[1])

    else:
        pass

    return data


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
            Nhigh = -Nex[0] + ML_sp.find_nearest(frequens, bandwidth[0][0])
            nsimdeci.append([0,Nhigh])

    j = 0
    while j != 1:
        i = 0
        while i != N:
            if (i+1)*AC_freq[j] + bandwidth[j + 1][i]/2 > truntime[1] or (i+1)*AC_freq[j] + bandwidth[j + 1][i]/2 >max(frequens):
                if (i + 1) * AC_freq[j] - bandwidth[j + 1][i]/2 < truntime[1]:
                    Nwindlow = -Nex[0] + ML_sp.find_nearest(frequens, (i + 1) * AC_freq[j] - bandwidth[j + 1][i]/2)
                    Nwindhigh = Nsimlength  # this will probs be some arbitary number/freq
                    #freqx.append(frequens[Nwindlow:Nwindhigh])
                    nsimdeci.append([Nwindlow, Nwindhigh])
                i = N
            else:
                Nwindlow = -Nex[0] + ML_sp.find_nearest(frequens, (i+1)*AC_freq[j] - bandwidth[j + 1][i]/2)
                Nwindhigh = -Nex[0] + ML_sp.find_nearest(frequens, (i+1)*AC_freq[j] + bandwidth[j + 1][i]/2)
                #freqx.append(frequens[Nwindlow:Nwindhigh])
                nsimdeci.append([Nwindlow,Nwindhigh])
                i += 1
        j += 1

    return nsimdeci,frequens

def mecwritter2(varibles,samplertype):

    var = []
    for i in range(len(varibles)):
        # for gaussian x = [mean,std,input,repeat]
        # for standard dev x = x = [resultion,low,high,log scale,input,repeat]
        """Addscan type in here"""
        if samplertype == 'normaldist':
            x = [varibles[i][0], varibles[i][1], None, 0, varibles[i][2], varibles[i][3]]
        elif samplertype == 'linear':
            x = [varibles[i][0], varibles[i][1], varibles[i][2], varibles[i][3], varibles[i][4], varibles[i][5]]
        elif samplertype == 'unirand':
            # min value, max value , log, value,repeat
            x = [None, varibles[i][0], varibles[i][1], varibles[i][2], varibles[i][3], varibles[i][4]]
        else:
            print('wrong sampler type inputed')
            exit()

        var.append(x)

    var = pd.DataFrame(var, columns=None)


def mecreader(Input_file):

    data = read_csv(Input_file, sep='    ', index_col=False, header=None, comment='!')

    # This needs to be moved till after CMA-Settings
    # Exp_data = Exp_data[0].str.split('  ',expand=True).astype('float')# \t for tab sep files # '  ' for mecsim

    return data

def functionholder(var):
    funcvar_holder = []
    # issolates up the functional parameters
    i = 0
    while i != len(var.iloc[:][0]):
        if var.iloc[i][4] == 41 or var.iloc[i][4] == 42:
            x = var.iloc[i].values
            x = np.append(x, i + 1)  # adds the varible column data to the area we need
            funcvar_holder.append(x)

        elif var.iloc[i][4] == 43: # varied potential between two other varibles
            x = var.iloc[i].values
            x = np.append(x, i + 1)  # adds the varible column data to the area we need
            funcvar_holder.append(x)
        elif var.iloc[i][4] == 44: # varied scalar value of another parameter
            x = var.iloc[i].values
            x = np.append(x, i + 1)  # adds the varible column data to the area we need
            funcvar_holder.append(x)

        else:
            pass
        i += 1

    return funcvar_holder

# Automaticly identifies if AC or DC methods should be used then allocates it
def ACDC_method(AC_amp):
    S = np.sum(AC_amp)
    DCAC_method = [0, 0, 0, 0]  # here to assign the fitting method and ACDC method

    # Optimization settings
    # op_settings = [0,0]     #prealocation
    DCAC_method[1] = 1  # sets the fitting function with {0 = absfit, 1 = %fit}
    DCAC_method[2] = 1  # Sets the number of cores
    DCAC_method[3] = None  # Sets the decimation number

    if S != 0:
        DCAC_method[0] = 1

    else:

        DCAC_method[0] = 0

    return DCAC_method


# returns AC frequency Line Number in MECSim input
def AC_info(data, spaces):
    ii = 1  # counter
    x = np.zeros((spaces[4], 2))  # preallocates a AC bin
    while ii != spaces[4] + 1:  # extracts all # of AC signals

        x[ii - 1, :] = data.iloc[spaces[0] + 1 + ii][0].split(
            ',')  # takes each AC input and splits frequency and amp
        ii += 1

    AC_freq = x[:, 1]  # sorts frequency into lowest to highest
    AC_amp = x[:, 0]  # Sum of all the AC signals

    return AC_freq, AC_amp

# Gets out the starting capacitance
def capacatance0(data,spaces,var):

    capacatance = []
    cap = False
    x = var.iloc[:][4].values

    i = 0
    while i != len(x):
        if 60 > int(x[i]) > 50:
            cap = True
        else:
            pass
        i += 1

    Csol = int(data.iloc[spaces[1]-1][0])

    i = 0
    while i != int(data.iloc[spaces[1]+Csol][0]) + 1:
        capacatance.append(float(data.iloc[spaces[1]+Csol +2+i][0]))
        i += 1

    capacatance1 = [cap, capacatance]

    return capacatance1


# something to load MASTER.file (data is a pandas dataframe)
def ModMec_form(data, spaces):
    # extracts values from data
    MECSettings, Numberin, PReacpara, PreacMech, PCapaci, PDiffvalues, PACSignal, PEModown \
        = input_sep(data, spaces)

    # turns values into
    Reacpara, reacMech, Capaci, Diffvalues, ACSignal, EModown, Currenttot, \
    Error_Code = pre2post(PReacpara, PreacMech, \
                          PCapaci, PDiffvalues, PACSignal, PEModown)

    return MECSettings, Numberin, Reacpara, reacMech, Capaci, Diffvalues, ACSignal, EModown, Currenttot, Error_Code


def pre2post(PReacpara, PreacMech, PCapaci, PDiffvalues, PACSignal, PEModown):
    # Outputs
    Error_Code = 0

    # Inputs
    # header file
    nstoremax = 10000000

    nrmax = 30
    nsigmax = 30
    nrmaxtot = 100
    nmaxErev = 10000
    nsp = 30
    nmaxcapcoeff = 4

    # static files
    Currenttot = np.zeros(nstoremax)  # the plus ones are due to fortran starting at zero
    EModown = np.zeros(nmaxErev + 1)
    ACSignal = np.zeros((nsigmax + 1, 2))
    Diffvalues = np.zeros((nrmax + 1, 3))
    reacMech = np.zeros((nsp + 1, nrmax + 1))
    Reacpara = np.zeros((nsp + 1, 4 + 1))
    Capaci = np.zeros(nmaxcapcoeff + 2)

    """"Formater for Numberin"""

    """Inputs values loaded into a static varibles for fortran uses lists"""
    # EModown
    i = 0
    while i != nmaxErev + 1:
        if i < len(PEModown):
            EModown[i] = PEModown[i]
            i += 1
        else:
            i = nmaxErev + 1

    # Capaci
    i = 0
    while i != nmaxcapcoeff + 2:
        if i < len(PCapaci):
            Capaci[i] = PCapaci[i]
            i += 1
        else:
            i = nmaxcapcoeff + 2

    # ACSignal
    i = 0
    while i != nsigmax + 1:
        if i < len(PACSignal):
            ACSignal[i, :] = PACSignal[i]
            i += 1
        else:
            i = nsigmax + 1

    # Diffvalues
    i = 0
    while i != nrmax + 1:
        if i < len(PDiffvalues):
            Diffvalues[i] = PDiffvalues[i]
            i += 1
        else:
            i = nrmax + 1

    # Reaction Parameters
    i = 0
    while i != nsp + 1:
        if i < len(PReacpara):
            Reacpara[i] = PReacpara[i]
            i += 1
        else:
            i = nsp + 1

    # reactionMech
    i = 0
    while i != nsp + 1:
        if i < len(PreacMech):
            j = 0
            while j < len(PreacMech[0]):
                reacMech[i, j] = PreacMech[i][j]
                j += 1
            i += 1
        else:
            i = nsp + 1

    return Reacpara, reacMech, Capaci, Diffvalues, ACSignal, EModown, Currenttot, Error_Code


# extracts data from Mecsim file  (data is a pandas dataframe)
def input_sep(data, spaces):
    # predefine input arrays
    # Numberin = [2**datapoints, NEVramp + 2, ACsource, Nspecies, NCap+1, Nreactions]

    # Set up parameters
    PReacpara = []
    PreacMech = []
    PCapaci = []
    PDiffvalues = []  # diff [conc,diff,surf]
    Numberin = [0, 0, 0, 0, 0, 0]
    PACSignal = []  # should be size (Nac, 2) [Amp,freq]
    PEModown = []  # [AdEst, AdEend, E_rev1, E_rev2] # should be size 4 + inf
    MECSettings = []  # should be size 32

    # set up parameters
    NVramp = int(data.iloc[19][0])  # gets number of NVramp

    # Numberin
    Numberin[0] = 2 ** float(data.iloc[6][0])  # Ndatapoint
    Numberin[1] = int(data.iloc[19][0]) + 2  # Nev
    Numberin[2] = int(data.iloc[33 + NVramp][0])  # NAC
    Numberin[3] = int(data.iloc[34 + NVramp + Numberin[2]][0])  # Nspecies
    Numberin[4] = int(data.iloc[35 + NVramp + Numberin[2] + Numberin[3]][0]) + 1  # Ncap+1
    Numberin[5] = len(data[:][0]) - (37 + NVramp + Numberin[2] + Numberin[3] + Numberin[4])  # Nmech

    # Extract Data into MECSettings
    """MECSettings = [Temp,Resistance, E_start, E_rev, Ncyc, scanrate, datapoints, 
               Digipotcompad, outputtype, ECtype, pre_equalswitch, fixtimestep,
               Nfixedsteps, beta, Dstar_min, maxvoltstep, timeres, debugout, 
               Advolt, NEVramp,Geotype, planararea, Nsphere, sphererad, Ncyilinders,
               rad_cylinder, cylLen, spacialres, RDErad, RDErots, RDEkinvis, Epzc]"""
    i = 0
    while i != len(data.iloc[:][0]):
        # thing to skip the boring shit

        if i == 19:  # constant value of NEVramp
            MECSettings.append(float(data.iloc[i][0]))
            PEModown.append(float(data.iloc[i + 1][0]))  # gets mod V ramp start
            PEModown.append(float(data.iloc[i + 2][0]))  # gets mod V ramp end
            j = 0
            while j != NVramp:
                PEModown.append(float(data.iloc[i + 2 + j + 1][0]))
                j += 1

            i = int(i + 2 + NVramp)

        elif i == 19 + 2 + NVramp + 12:  # kinematics point ready for AC

            j = 0
            while j != Numberin[2]:
                x = []  # holder for AMP,freq
                y = data.iloc[i + 1 + j][0].split(',')
                x.append(float(y[0]))  # get amp
                x.append(float(y[1]))  # get freq
                PACSignal.append(x)
                j += 1
            i = i + 1 + Numberin[2]


        elif i == spaces[1]:  # Diffusion values to be put into the crap
            j = 0
            while j != Numberin[3]:
                x = []  # holder for AMP,freq
                y = data.iloc[spaces[1] + j][0].split(',')
                x.append(float(y[0]))  # get Conc
                x.append(float(y[1]))  # get Diff
                x.append(float(y[2]))  # get surf
                PDiffvalues.append(x)
                j += 1
            i = i - 1 + Numberin[3]

        elif i == spaces[1] + Numberin[3]:  # Cap values to be put into the crap
            PCapaci.append(float(data.iloc[spaces[1] + Numberin[3]][0]))
            MECSettings.append(float(data.iloc[i + 1][0]))
            j = 0
            while j != Numberin[4]:
                PCapaci.append(float(data.iloc[spaces[1] + 2 + j + Numberin[3]][0]))
                j += 1
            i = spaces[2] - 1

        elif i == spaces[2]:  # Kinetic mechanism

            j = 0
            while j != Numberin[5]:
                x1 = []  # Mechanism
                x2 = []  # mech para
                y = data.iloc[spaces[2] + j][0].split(',')

                k = 0
                while k != Numberin[3] + 1:
                    x1.append(float(y[k]))
                    k += 1
                PreacMech.append(x1)
                while k != Numberin[3] + 6:
                    x2.append(float(y[k]))
                    k += 1
                PReacpara.append(x2)
                j += 1
            i = len(data.iloc[:][0]) - 1

        else:

            MECSettings.append(float(data.iloc[i][0]))

        i += 1

    """example    
    reacMech1 = [[0,-1,1,0],[2,0,-1,1]]
    Reacpara1 = [[2,10 ,0,1000,0.5],[2,2 ,0.1,2000,0.5]]
    Capacipre = [NCap, cap0, cap1, cap2, cap3, cap4]
    Diffvaluespre = [[Conc1,Diff1,surf1],[Conc2,Diff2,surf2],[Conc3,Diff3,surf3]]
    ACSignalpre = [[A1, freq1],[A2, freq2]]
    EModownpre = [AdEst, AdEend, E_rev1, E_rev2]"""

    return MECSettings, Numberin, PReacpara, PreacMech, PCapaci, PDiffvalues, PACSignal, PEModown

def outputsetterDT(data):

    # extracts important parameters
    Estart = float(data.iloc[2])
    Erev = float(data.iloc[3])
    Ncycles = float(data.iloc[4])
    Scanrate = float(data.iloc[5])
    #dpoints = 2**int(data.iloc[6])

    DeltaT = (Ncycles*2*abs(Erev - Estart)/Scanrate)

    return DeltaT

def outputsetter(data,AC_freq, AC_amp):

    # extracts important parameters
    Estart = float(data.iloc[2])
    Erev = float(data.iloc[3])
    Ncycles = float(data.iloc[4])
    Scanrate = float(data.iloc[5])
    dpoints = 2**int(data.iloc[6])

    Timetotal = Ncycles*2*abs(Erev - Estart)/Scanrate

    #calculate time vector
    Time = np.linspace(0,Timetotal,dpoints)

    #calculate voltage
    # calculate DC
    for i in range(int(Ncycles)):
        DC1 = np.linspace(Estart,Erev,int(dpoints/(2*Ncycles)))
        DC2 = np.linspace(Erev, Estart, int(dpoints / (2 * Ncycles)))
        if i == 0:
            DC = np.append(DC1,DC2)
        else:
            x = np.append(DC1, DC2)
            DC = np.append(DC,x)

    voltage = DC

    for i in range(len(AC_amp)):
        voltage = voltage + (AC_amp[i]/1000)*np.sin(2*np.pi*Time*AC_freq[i])

    return Time, voltage

def MECoutsetter(data,AC_freq, AC_amp):

    # gets values and shit
    Estart = float(data.iloc[2])
    Erev = float(data.iloc[3])
    Ncycles = int(data.iloc[4])
    Scanrate = float(data.iloc[5])

    s = 'cvsin_type_1  \n'
    s += 'Start(mV):      ' + format_e_out(Estart) +'\n'
    s += 'End(mV):        ' + format_e_out(Estart) +'\n'
    if Estart > Erev:
        s += 'Max(mV):        ' + format_e_out(Estart) +'\n'
        s += 'Min(mV):        ' + format_e_out(Erev) +'\n'
    elif Erev > Estart:
        s += 'Max(mV):        ' + format_e_out(Erev) +'\n'
        s += 'Min(mV):        ' + format_e_out(Estart) +'\n'
    else:
        print('MECSim does not take flat inputs of this form')
        exit()

    # prints the sine wave propities
    for i in range(1,9):
        count = '%i' % i
        if i < len(AC_amp) + 1:
            s += 'Freq' + count + '(Hz):      '+ format_e_out(AC_freq[i-1]) +'\n'
            s += 'Amp' + count + '(mV):       ' + format_e_out(AC_amp[i-1]) + '\n'
            #s += 'Phase' + count + '(deg):    0.000000E+00\n'
        #else:
            #s += 'Freq' + count + '(Hz):      1.000000E+00\n'
            #s += 'Amp' + count + '(mV):       0.000000E+00\n'
            #s += 'Phase' + count + '(deg):    0.000000E+00\n'

    s += 'Type(0-4):      4\n'
    s += 'Rate(mV/s):     ' + format_e_out(Scanrate*1000) +'\n'
    s += 'Direction:      c ' + '\n'
    s += 'Scans:          ' +str(Ncycles)  +'\n'
    s += 'Average:        19\n'
    s += 'Gain(0-4):      1 \n'
    s += 'Initial(mV):    ' + format_e_out(Estart) +'\n'
    s += 'Initial(mS):    1.000000E+04\n'
    s += 'Pre(mV):        0.000000E+00\nPre(ms):        0.000000E+00\n'
    s += 'Post(mV):       0.000000E+00\nPost(ms):       0.000000E+00\n'

    return s

# some function for passing the experimental setings to the sql database
def sqlexpsettings(serverdata,data, spaces, nondim, AutoNharm,bandwidth,ReactionMech,windowing,guass_std,inputtxt):
    #connect to sql server
    dic = sqlexpdicset(data, spaces)
    dic.update({"AutoNharm": AutoNharm})
    dic.update({"Nonedimcurr":nondim})
    dic.update({"IFT_windowing": windowing}) # windowing classification
    dic.update({"WinGuass_std": guass_std}) # gussian value of window
    dic.update({"inputfile":inputtxt})
    # set up for bandwdth
    x = []
    x.append(bandwidth[0][0])   # DC bandwidth
    for values in bandwidth[1]:
        x.append(values)
    dic.update({"bandwidth": x})
    dic.update({"ReactionMech": ReactionMech}) # here to include the reaction mech into experimental setting

    try:

        connection = psycopg2.connect(user=serverdata[0],
                                      password=serverdata[1],
                                      host=serverdata[2],
                                      port=serverdata[3],
                                      database=serverdata[4])

        cursor = connection.cursor()

        cursor.execute("""SELECT * FROM "ExperimentalSettings" """)

        if type(cursor.fetchone()) == type(None):    # this is to check if there are no rows at all in the database if None returned its empty
            EXPsetrow = 1
            dic.update({"EXPsetrow": EXPsetrow})

        else:
            cursor.execute("""SELECT "EXPsetrow" FROM "ExperimentalSettings" WHERE "EXPsetrow" = (SELECT MAX ("EXPsetrow") FROM "ExperimentalSettings") """)
            x = cursor.fetchall()[0][0]     # This extracts a lot of
            EXPsetrow = x + 1
            dic.update({"EXPsetrow": EXPsetrow})

        qinput = """INSERT INTO "ExperimentalSettings" VALUES( %(Estart)s, %(Erev)s, %(cyclenum)s, %(Scanrate)s, %(datapoints)s
                    , %(GeometryType)s, %(PArea)s, %(Nac)s, %(ACFreq1)s, %(boolsim)s,
                      %(EXPsetrow)s,%(AutoNharm)s,CURRENT_TIMESTAMP,%(bandwidth)s,%(ReactionMech)s,
                     %(IFT_windowing)s, %(WinGuass_std)s ,null, %(inputfile)s ,%(Nonedimcurr)s)"""

        cursor.execute(qinput, dic)
        connection.commit()

    except (Exception, psycopg2.Error) as error:
        print("error,", error)
    finally:
        # This is needed for the
        if (connection):
            cursor.close()
            connection.close()

    # insert the values into sql

    return EXPsetrow

# Function for preallocation of the Rection ID for pooling
def sqlReacIDPreallo(serverdata,N,sqlExpinput,ReactionMech):

    try:
        connection = psycopg2.connect(user=serverdata[0],
                                      password=serverdata[1],
                                      host=serverdata[2],
                                      port=serverdata[3],
                                      database=serverdata[4])

        cursor = connection.cursor()

        cursor.execute("""SELECT * FROM "Simulatedparameters" """)

        # checks to see start point of the RECT_ID
        if type(cursor.fetchone()) == type(None):    # this is to check if there are no rows at all in the database if None returned its empty
            Reaction_ID = 1

        else:
            cursor.execute("""SELECT "Reaction_ID" FROM "Simulatedparameters" WHERE "Reaction_ID" = (SELECT MAX ("Reaction_ID") FROM "Simulatedparameters") """)
            x = cursor.fetchall()[0][0]     # This extracts a lot of
            Reaction_ID = x + 1

        # Goes through and adds the ID to the database for the two main tables
        for i in range(N):
            qinput = """INSERT INTO "Simulatedparameters"("Reaction_ID","EXPsetrow") VALUES(%s,%s)"""
            cursor.execute(qinput, (Reaction_ID+i,sqlExpinput,))
            qinput = """INSERT INTO "HarmTab"("Reaction_ID","EXPsetrow") VALUES(%s,%s)"""
            cursor.execute(qinput, (Reaction_ID+i,sqlExpinput,))
            ReactionMech
        connection.commit()

        dic = {}
        dic.update({"React_IDrange":[int(Reaction_ID),int(Reaction_ID+i)]})
        dic.update({"EXPsetrow":int(sqlExpinput)})
        qinput = """UPDATE "ExperimentalSettings" SET "React_IDrange" = %(React_IDrange)s WHERE "EXPsetrow" = %(EXPsetrow)s """
        cursor.execute(qinput, dic)
        connection.commit()

    except (Exception, psycopg2.Error) as error:
        print("error,", error)
    finally:
        # This is needed for the
        if (connection):
            cursor.close()
            connection.close()

    # insert the values into sql

    return Reaction_ID


# some function forsetting up the dictionary
def sqlexpdicset(data, spaces):
    dic = {}
    # Estart
    dic.update({"Estart": float(data.iloc[2])})
    # Erev
    dic.update({"Erev": float(data.iloc[3])})
    # cyclenum
    dic.update({"cyclenum": int(data.iloc[4])})
    # Scanrate (V/s)
    dic.update({"Scanrate": float(data.iloc[5])})
    # datapoints
    dic.update({"datapoints": 2**float(data.iloc[6])})
    # GeometryType
    dic.update({"GeometryType": int(data.iloc[spaces[0]-10])})
    # PArea
    dic.update({"PArea": float(data.iloc[int(spaces[0] - 9)])})
    # Nac
    dic.update({"Nac":int(data.iloc[int(spaces[0] + 1)])})
    #ACFreq1 real[]
    x = data.iloc[int(spaces[0] + 2)][0].split(",")
    acfreq = [float(x[0]),float(x[1])]
    dic.update({"ACFreq1": acfreq})
    # boolsim
    dic.update({"boolsim":True})
    # timeused REMOVED DUE TO UPDATES
    #dic.update({"timeused": Timesim})
    # Voltageused
    #dic.update({"Voltageused": voltagesim})
    # pass EXPsetrow and set that up at the connector section

    return dic

# function to get around python printing numbers truncated and worng0
def format_e_out(n):

    a = '%E' % n
    fucker = a.split('E')[0].rstrip('0')
    for i in range(len(fucker),8):
        fucker += '0'
    fucker +='E' + a.split('E')[1]

    return  fucker

# writes the MECsim output to a txt file compadable with
def outputwriter(filename,i,startpart,voltage,Scurr,timev):

    counter = '%i' %i
    name = filename + counter+'.txt'
    timer = time.time()
    f = open(name, 'w')
    f.write(startpart)
    timer = time.time()
    for i in range(len(Scurr)):
        s = format_e_out(voltage[i]) + '   ' + format_e_out(Scurr[i]) + '   ' \
            + format_e_out(timev[i]) + '\n'
        f.write(s)

    print(time.time() - timer)

    return

def MECSiminpwriter(filename,i, val_in, **kwargs):  # (data, var, val_in, spaces, DCAC_method, Exp_data,harm_weights):

    # Extracts the varibles
    data = kwargs.get('data')
    var = kwargs.get('var')
    spaces = kwargs.get('spaces')
    #Exp_data = kwargs.get('Exp_data')
    scalvar = kwargs.get('scalvar')
    DCAC_method = kwargs.get('DCAC_method')
    funcvar = kwargs.get('funcvar')
    funcvar_holder = kwargs.get('funcvar_holder')
    cap_series = kwargs.get('cap_series')
    # extracts and removes the counter
    #counter = val_in[-1]
    #val_in = val_in[0:-1]
    #Ncore = DCAC_method[2]

    data = Var2data(data, var, val_in, spaces)  # turns the varibles into data

    # sets up for scalar dependant varibles
    if scalvar[0][0] != 0:
        data = Scalvar2data(scalvar, val_in, data, spaces)
    else:
        pass

    if funcvar[0][0] != 0:
        funclist = listin_func(val_in, funcvar, funcvar_holder)
        data = funcvar2data(funclist, spaces, data)
    else:
        pass

    # checks to see if capacitance is present


    if cap_series[0]:

        data = cap2data(val_in, var, spaces, data,cap_series[1])

    counter = '%i' % i
    name = filename + counter + '.inp'

    MECSimwriter(name,data)

    return

def settings_writer(filename,varibles, samplertype, scalvar, funcvar,samples, Cnoise, Noisestore):

    f = open(filename, 'w')

    f.write('Date: %s\n' % (datetime.datetime.today().strftime('%d-%m-%Y')))
    f.write(samplertype + '\n\n')
    f.write('varibles:\n')
    for items in varibles:
        for values in items:
            f.write(str(values)+ '\t')
        f.write('\n')
    f.write('\n\n')

    f.write('scalar varibles\n')
    for items in scalvar:
        for values in items:
            f.write(str(values) + '\t')
        f.write('\n')
    f.write('\n\n')

    f.write('functional varibles\n')
    for items in funcvar:
        for values in items:
            f.write(str(values) + '\t')
        f.write('\n')
    f.write('\n\n')

    f.write('guassian current noise used: %f\n\n' %Cnoise)

    f.write('sample values used\n')
    for items in samples:
        for values in items:
            f.write(str(values) + '\t')
        f.write('\n')
    f.write('\n\n')

    f.write('Real mean and std: (mean \tstd)\n')
    for i in range(len(samples[0])):
        x = []
        for items in samples:
            x.append(items[i])
        mean = np.mean(x)
        std = np.std(x)
        f.write(str(mean) + '\t' +str(std) +'\n')

    f.write('\nNoise used for each independant total current output (A)\n')
    for i in range(len(Noisestore)):
        f.write(str(Noisestore[i]) +'\n')

    f.close()

    return

def singleharmplot(outputfname,hil_store,Simtime,bandwidth):

    os.makedirs(outputfname)

    N = len(bandwidth[0])

    pererror = []

    # plot harmonics
    i = 0
    while i != N + 1:

        if i == 0:
            textstr = 'DC Signal'
        elif i == 1:
            textstr = '1st Harmonic'
        elif i == 2:
            textstr = '2nd Harmonic'
        elif i == 3:
            textstr = '3rd Harmonic'
        else:
            textstr = '%ith Harmonic'   %i

        plt.figure()
        fig, ax = plt.subplots()
        plt.plot(Simtime, hil_store[i, :], color='r',label='Simulated',linestyle='-.')
        plt.ylabel('Current (Amps)')
        plt.xlabel('Time (sec)')
        plt.text(0.40, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                 verticalalignment='top', size=10, bbox=dict(facecolor='white', alpha=0.5, boxstyle="square"))
        plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        ax.legend(loc = 'upper right')
        s = '%s/Harmonic%i' % (outputfname, i)
        plt.savefig(s, bbox_inches='tight')
        plt.close()

        # error as function of time
        i += 1

    return

def windowfunctioning(windowing,fft_freq,bandwidth,guass_std,AC_freq,Nharm):

    filter_hold = []

    if windowing == "Convolutional_Guass":

        Convguass = Wint.analitical_RGguassconv_fund(fft_freq, bandwidth[1][0], guass_std * bandwidth[0][0])
        filter_hold.append(Convguass)

        if Nharm != 0:
            for NNNN in range(Nharm+1):
                # analytical solution to guass rec convultion
                #print(NNNN * AC_freq)
                Convguass = Wint.analitical_RGguassconv(fft_freq, bandwidth[1][NNNN], (NNNN+1) * AC_freq,
                                                        guass_std * bandwidth[1][NNNN])
                filter_hold.append(Convguass)

        else:

            Convguass = Wint.analitical_RGguassconv(fft_freq, bandwidth[1][1], AC_freq,
                                                    guass_std * bandwidth[1][1])
            filter_hold.append(Convguass)

    elif windowing == "Rect":

        rect = Wint.square_window_fund(fft_freq, bandwidth[0][1])
        filter_hold.append(rect)
        if Nharm != 0:
            for NNNN in range(1, Nharm+1):
                rect = Wint.square_window(fft_freq, NNNN * AC_freq, bandwidth[1][NNNN-1])
                filter_hold.append(rect)
        else:
            rect = Wint.square_window(fft_freq, AC_freq, bandwidth[1][0])
            filter_hold.append(rect)

    else:
        print("Error: no correct filter windowing inputted")
        exit()

    return filter_hold

