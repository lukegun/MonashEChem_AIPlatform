# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:11:50 2020

@author: luke
"""

import sys
import os

# taken from the biomec git and repurposed to work
def inputwritter(filename,optimisation,logicm,data,expfile,var,winfunc,settingsdic,MEC_set,trunstime):
    # inputs seperators
    breaker1 = "!Settings   (value,min,max,log sensitivity, parameter code, repeat line)"
    breaker2 = "MECSim settings   ! DO NOT REMOVE THESE SEPERATORS"
    breaker3 = "Experimental settings ! {NameEX.txt} Repet for triplicates effects method Important do not remove this line"

    #################### PART THAT TELLS THE CODE WHAT TO DO #####################

    # header settings
    tot = "STANDARD"
    # sets up optimisation method
    correctinput = False
    while not correctinput:
        #optimisation = str(input("Please input optimisation method (CMAES or ADMCMC): "))
        if optimisation == "CMAES" or optimisation == "ADMCMC":
            correctinput = True
        else:
            print("Incorrect optimisation input")

    correctinput = False
    while not correctinput:
        #print("List of logic methods:\nHarmPerFit\nBayes_ExpHarmPerFit\nTCDS\nFTC\nLog10FTC ")
        #logicm = str(input("Please input logic method : "))
        if logicm == "HarmPerFit" or logicm == "Baye_HarmPerFit" or logicm == "Bayes_ExpHarmPerFit" or logicm == "TCDS" or logicm == "Log10FTC" or logicm == "FTC":
            correctinput = True
        else:
            print("Incorrect optimisation input, please input exactly from above list")

    header = [tot, logicm, optimisation]

    ####################### Start OF OPT SETTINGS ################################
    scaledpara = MEC_set.get("scalvar")
    funcparas = MEC_set.get("funcvar")
    harmband = MEC_set.get("bandwidth")
    harmwieght = MEC_set.get('harm_weights')
    #makes harmweights in right format
    h = []
    x = [harmwieght[0]]
    for i in range(len(harmband[0])):
        x.append(0)
    h.append(x)
    x = []
    for i in range(len(harmband[0])):
        x.append(harmwieght[i + 1])
    h.append(x)

    Nharm = len(harmband[0])
    Nac = len(MEC_set.get("AC_freq"))
    Ncore = MEC_set.get("op_settings")[1]
    gensetting = optsettings(header,var,winfunc,scaledpara,funcparas,Nac,Nharm,harmband,h,1,trunstime[0],trunstime[1],Ncore)

    # custom optimisation settings
    if optimisation == "CMAES":
        Ncore = settingsdic.get("Ncore")
        Ndata = settingsdic.get("Ndata")
        tolx = settingsdic.get("tolx")
        sigma = settingsdic.get("sigma")
        outputlist = CMAsettings(Ncore,Ndata,tolx,sigma)
    elif optimisation == "ADMCMC":
        Ncore = settingsdic.get("Ncore")
        Ndata = settingsdic.get("Ndata")
        Nprior = settingsdic.get("Nprior")
        chaintot = settingsdic.get("chaintot")
        burnin = settingsdic.get("burnin")
        outputlist = ADMCMCsettings(Ndata,Ncore,Nprior,chaintot,burnin)
    else:
        print("ERROR in assigning header settings, TERMINATING PROGRAM")
        sys.exit()

    file_startpart = gensetting + outputlist

    ####################### PART THAT TAKES IN MECFILE AND EXP ###################

    # Takes the input file for the simulation
    x = data.values
    MECfile = []
    for values in x:
        MECfile.append(values[0])

    #print(MECfile)
    """mecreal = False
    while not mecreal:
        MECfilename = str(input("Please input MECSim input file name: "))
        # check to see if the MECsim file is real
        if os.path.exists(MECfilename):
            mecreal = True
            with open(MECfilename) as f:
                MECfile = f.read().splitlines()
        else:
            print("MECSim file was incorrect and could not be found, try again.")"""

    Nexp = len(expfile) #int(input("Number of experimental files you are comparing the simulation to: "))
    #print("put something here so it works with logic FIX")
    experimentalfile = []
    """print(
        "Please put the file input location relative to \nwhere you're running MECSim analytics package\neg. NameEX.txt or file/NameEX.txt ")"""
    for x in expfile:
        #x = str(input("Experimental filename " + str(i + 1) + ": "))
        experimentalfile.append(x)

    # sets up for the bottom part
    file_endpart = ["\n"] + [breaker2] + MECfile + ["\n"] + [breaker3] + experimentalfile

    filetot = [breaker1] + file_startpart + file_endpart

    ######################## PART THAT WRITES THE FILE ###########################
    """print("settings have been input succsessfully")
    filename = str(
        input("Please input name of text file these settings will be written to? (include .txt at the end): "))"""

    filewritter(filename, filetot)

    #print("process has been completed with settings saved to text file " + filename)



def optsettings(header,var,winfunc,scaledpara,funcparas,Nac,Nharm,harmband,harmwieght,trun,mintime,maxtime,Ncore):
    
    #Npara = int(input("Please input number of parameters to optimise for: "))
    
    paralist = paraopt(header,var)
    
    scalfunclist = scalarfuncvaribles(scaledpara,funcparas)
    
    if header[1] == "TCDS":

        harmfilter = ["0 ! windowing function (Zero for square, 1 for CG windowing;std = 0.1 Hz recomended)"]
        
        bDCside = " ! Bandwidth Fundimental (BANDWIDTH FIRST)"
        wDCside = " ! Fundimental weights"
        bharmside = " ! Harmonic Bandwidth REPEAT (lowest freq to highest)"
        wharmside = " ! Harmonic Weights REPEAT (lowest freq to highest)"
        
        harmsetting = ["1,0" + bDCside,"0,0" + bharmside,"1,0" + wDCside,"0,0" + wharmside]
    else:

        derp = " ! windowing function (Zero for square, 1 for CG windowing;std = 0.1 Hz recomended)"
        correct = True
        #print("Please input windowing function for harmonics\n0 = square window (Simple but more ringing)\n1 = Guassian Convolutional (Simple but more ringing)\n")
        while correct:
            #logicm = int(input("Please input windowing function number : "))
            if winfunc[0] == 0 or winfunc[0] == 1:
                correct = False
                if winfunc[0] == 1:
                    #fl = float(input("Please input guassian std (0.1 strongly recommended) : "))
                    x = '%i, %.4f' %(winfunc[0], winfunc[1])
                else:
                    x = str(winfunc[0])
            else:
                print("incorrect input try again")

        harmfilter = [x + derp]

        harmsetting = harmonichandler(Nac,Nharm,harmband,harmwieght)
    
    # Takes in logic for frequency domain shit 
    trunsettings = timetrun(trun,mintime,maxtime,Ncore)

    s = "%s %s %s ! Header declaration (tot Logic Method)" %(header[0],header[1], header[2])
    headertot = [s]
    
    generaloutputs = headertot + paralist + scalfunclist + harmfilter + harmsetting + trunsettings
    
    return generaloutputs

def paraopt(header,var):
    
    """print("\nList of parameters and relavent code:\n")
    print("Resistance: 11\nKinematic Viscosity: 12\nExperimental noise (TCDS only): 13\n"\
          + "Concentration: 21\n Diffussion coefficent: 22\nForward Reaction rate: 31\n"\
          + "Backward Reaction rate: 32\nFormal Potential: 33\nElectron Transfer Rate: 34\n"\
          + "Alpha or Lambda: 35\nEquilibrium Constant magnitude (func): 41\nEquilibrium Constant Theta (func): 42\n"\
          + "Capacitance constants C0-C4: 51-55\nCapacitance Scalar Multiple: 56\n\n")"""
    
    s = "%s\t\t! Number of varibles" %int(var.shape[0])
    paramterlist = [s]
    for i in range(var.shape[0]):
        #s = "\nParameter #%i" %(i+1)
        #print(s)
        
        code = var.iloc[i][4] #int(input("Please give code of parameter you want to optimise: "))
        if code>20 and code<45:
            repeat = var.iloc[i][5]#"""int(input("Please input which repeat line the parameter \n"\
                               #"is on in MECSim file (Start at 1): "))"""
        else:
            repeat = 1
            
        if header[2] == "CMAES":
            logs = var.iloc[i][3]#int(input("is this parameter optimised in a log scale (yes = 1 or no = 0): "))
        else:
            logs = 0

        correctinput = False
        while not correctinput:
            xmed = var.iloc[i][0]#float(input("Please input median parameter value: "))
            xsmall = var.iloc[i][1]#float(input("Please input smallest parameter range: "))
            xlarge = var.iloc[i][2]#float(input("Please input largest parameter range: "))
            if xsmall < xmed and xmed < xlarge:
                correctinput = True
            else:
                print(xmed,xsmall,xlarge)
                print("\n Parameters range is not correct, please redo order of small, med large is correct")
                exit()
        
        xlist = "%s, %s, %s, %i, %i, %i" %(format_e(xmed),format_e(xsmall),format_e(xlarge),logs,code,repeat)
        
        paramterlist.append(xlist)
        
    return paramterlist

def scalarfuncvaribles(scaledpara,funcparas):
    
    Ns = scaledpara[0][0] #int(input("Please input number of scaled varibles: "))
    s = "%i\t\t! number of scaled varibles" %Ns
    Nscallist = [s]
    
    side = " \t! scaling reletive varibles (scaling factor, varible number, parameter code, repeat line)"
    for i in range(Ns):
        Npara = scaledpara[i + 1][1] # int(input("Please input which parameter is being scaled in the input parameter (Starts at one): "))
        scal = scaledpara[i + 1][0]  #float(input("Please input the scalar value: "))
        Scode = scaledpara[i + 1][2] #int(input("Please input output parameter varible code: "))
        Srepeat = scaledpara[i + 1][3] # int(input("Please input output parameter reapeat line in MECSim: "))
     
        s = "%.4f, %i, %i, %i" %(scal, Npara, Scode,Srepeat)
        Nscallist.append(s + side)
        
    # functional varibles settings
    Nf = funcparas[0][0] #int(input("Please input number of fuctional varibles: "))
    s = "%i\t\t! Number of functional scaling" %Nf
    Nscallist.append(s)
    
    side = " \t! functional parameter (function, varible number, paired value scalar, {paired value, varible row})"
    for i in range(Nf):
        Npara = funcparas[i + 1][0] #int(input("Please input function: "))
        scal = funcparas[i + 1][1]# int(input("Please input which parameter is being used in function (Starts at one): "))
        Scode = funcparas[i + 1][2]#int(input("Please input paired value scalar (use 0 if nothing): "))
        Srepeat = funcparas[i + 1][3]#float(input("Plese input paired value or paired varible input row: "))
        
        if Scode == 1:
            s = "%i, %i, %i, %i" %(Npara, scal, Scode,int(Srepeat))
        else:
            s = "%i, %i, %i, %s" %(Npara, scal, Scode,format_e(Srepeat))
        Nscallist.append(s + side)
        
        
    return Nscallist

def harmonichandler(Nac,Nharm,harmband,harmwieght):
    
    #Nac = int(input("Please input number of AC frequencies used in experimental data: "))
    #Nharm = int(input("Please input number of harmonics present (excluding DC): "))
    
    #DCband = float(input("Please input bandwidth of DC component: "))
    #DCwieght = float(input("Please input scalar weight of DC component: "))
    
    
    # sets up for the DC somponent
    DCbandhold = "%.2f" %harmband[0][0]
    DCwieghthold = "%.2f" %harmwieght[0][0]
    
    for i in range(Nharm - 1):
        DCbandhold += ", 0"
        DCwieghthold += ", 0"
        
    # 
    if Nac > 1:
        """#print("As you are using more then one AC frequency note that\n" +\
              "you will be asked to input multple sets of data with the\n" +\
              "these will be analyised by the code from LOWEST FREQUENCY TO "+\
              "HIGHEST FREQUENCY.")"""
        
    tothbandhold = []
    tothwieghthold = [] 
    
    for i in range(Nac):
        #harmband = float(input("Please input bandwidth of fundimental harmonic: "))
        #harmwieght = float(input("Please input scalar weight of fundimental harmonic: "))
        
        harmbandhold = "%.2f" %harmband[i+1][0]
        harmwieghthold = "%.2f" %harmwieght[i+1][0]
        
        correct_inp = False
        while not correct_inp:
        
            same = 1 #int(input("Are the harmonic weights and bandwidths the same for all harmonics? (0 = No; 1 = Yes): "))
            if same == 1:
                for j in range(Nharm-1):
                    harmbandhold += ", %.2f" %harmband[i+1][0]
                    harmwieghthold += ", %.2f" %harmwieght[i+1][0]
                    
                    correct_inp = True
                        
                """elif same == 0:
                for j in range(Nharm - 1):
                    hb = float(input("Please input bandwidth of the #%i harmonic: " %(j+1) ))
                    hw = float(input("Please input scalar weight of the #%i harmonic: " %(j+1)))
                    
                    harmbandhold += ", %.2f" %hb
                    harmwieghthold += ", %.2f" %hw
                    
                    correct_inp = True   """
            else:
                print("Incorrect input on same")
                
        tothbandhold.append(harmbandhold) 
        tothwieghthold.append(harmwieghthold) 
        
    #sets the side stings
    bDCside = " ! Bandwidth Fundimental (BANDWIDTH FIRST)"
    wDCside = " ! Fundimental weights"
    bharmside = " ! Harmonic Bandwidth REPEAT (lowest freq to highest)"
    wharmside = " ! Harmonic Weights REPEAT (lowest freq to highest)"
        
    # something to put this all in as an output
    # band first
    harmsetting = [DCbandhold + bDCside]
    for i in range(Nac):
        harmsetting.append(tothbandhold[i] + bharmside)
    
    # harmoni
    harmsetting.append(DCwieghthold + wDCside)
    for i in range(Nac):
        harmsetting.append(tothwieghthold[i] + wharmside)
        
        
    return harmsetting

# does truncation time with exception to frequency domain plus junk parameters
def timetrun(trun,mintime,maxtime,Ncore):
    
    timeend = "\t\t\t! Truncation points (sec) (0,MAX; for not applicable)"
    
    correct_inp = False
    while not correct_inp:
        #trun = int(input("Are you truncating the time series for analysis? (0 = No, 1 = Yes): "))
        # truncation time
        if trun == 1:
            """if logic == "Log10FTC" or logic == "FTC":
                mintime = float(input("Please input the minimum truncation Frequency (Hz): "))
                maxtime = float(input("Please input the maximum truncation Frequency (Hz): "))
            else:
                mintime = float(input("Please input the minimum truncation time (sec): "))
                maxtime = float(input("Please input the maximum truncation time (sec): "))"""
            
            s = "%.3f,%.3f" %(mintime,maxtime)
            correct_inp = True
            
        elif trun == 0:
            s = "0,MAX"               
            correct_inp = True
        
        else:
            print("ERROR: Incorrect previous input, retry")
        
    trunsettings = [s + timeend]
    
    # Fitting method used
    s = "1\t\t\t! Fitting method used  (0 = absdiff, 1 = %diff)"
    trunsettings.append(s)
    
    # Experimental input type
    #Ncore = int(input("Please input Experimental input type as number where (0 = MECSim, 1 = FTACV, 2=CHI): "))
    k = 1
    s = "%i\t\t\t! Experimental input type (0 = MECSim, 1 = FTACV, 2=CHI)" %k
    trunsettings.append(s)
    
    return trunsettings
    

def CMAsettings(Ncore,Ndata,tolx,sigma):
    
    settingslist = []
    
    # Number of multiprocesses
    #print("\nThe calculation is generally run accross multiple CPU processors to speed up calculations")
    #print("Warning for CMAES do not exceed int(3*log(#parameters)")
    #Ncore = int(input("Please input number of CPU processors: "))
    s = "%i\t\t\t! Number of cores to be used" %Ncore
    settingslist.append(s)
    
    # 2^N comparison points
    #print("\nTo speed up calculations only certain number of experimental data points are optimised of order of 2^N")
    #Ndata = int(input("Please input N in above statement: "))
    s = "%i\t\t\t! 2^N comparison data points per current" %Ndata
    settingslist.append(s)
    
    # tolx IMPORTANT TO EXPLAIN
    #print("\nTolx is the termination criterion for CMAES, it gives the level of of accuracy till it stops (low = 0.05, mid = 0.025, high = 0.01)")
    #tolx = float(input("Please input Tolx in above statement: "))
    s = "%.4f\t\t\t! tolx, value of x as %%*range/2 needed before fit is meet (0.05,0.025,0.01 recomended)"  %tolx
    settingslist.append(s)
    
    # initial sigma value as %*range (0.33 recomendanded)
    #print("\nIntial sigma for CMA-ES is the starting range as a ratio of starting scan range")
    #sigma = float(input("Please input sigma in above statement (0.33 is strongly recomended): "))
    s = "%.4f\t\t\t! initial sigma value as %%*range (0.33 recomendanded)"  %sigma
    settingslist.append(s)
    
    return settingslist

def ADMCMCsettings(Ndata,Ncore,Nprior,chaintot,burnin):
    
    settingslist = []
    
    # 2^N comparison points
    #print("\nTo speed up calculations only certain number of experimental data points are optimised of order of 2^N")
    #Ndata = int(input("Please input N in above statement: "))
    
    # number of overall chains
    #Nchain =str(input("\nPlease input number of MCMC chains desired (soft limit 4): "))
    
    Nchain = Ncore
    
    s = "%s\t\t\t! Number of cores to be used" %Ncore
    settingslist.append(s)
    s = "%s\t\t\t! 2^N comparison data points per current" %Ndata
    settingslist.append(s)  
    
    # prior sampling
    #print("\nIn ADMCMC prior sampling is inital run as part of the algorithm.")
    #Nprior = int(input("Please put initial prior sampling per varible (100-250 recomended): "))
    s = "%i\t\t\t! MCMC initial prior sampling per varible" %Nprior
    settingslist.append(s)
    
    # Number of trails in overall chain
    #print("\nThis is number of simulations for the entirly, remember that each chain = totchain/#")
    #chaintot = int(input("Please give the number of total simulations to run overall: "))
    s = "%i\t\t\t! number of trail for overall chain" %chaintot
    settingslist.append(s)
    
    # burnin period
    #print("\nAt the end of the MCMC calculation a percentage of the trails are thrown away, this is refered to as burnin")
    #burnin = float(input("Please put in the burnin % (%/100 so between 0-1): "))
    s = "%.4f\t\t\t! burnin period as ratio of chain legth (%%/100)" %burnin
    settingslist.append(s)
    
    # noise (mostly obsolte)
    sigma = "1.0e-6\t\t\t! noise (%%/100) for each frequency (lowest freq to highest)"
    settingslist.append(sigma)
    
    s = str(Nchain) + "\t\t\t! number of chains to run"
    settingslist.append(s)
    
    return settingslist

def filewritter(filename, filetot):
    
     f = open(filename,"w")
     for lines in filetot:
         if lines != "\n":
             f.write(lines + "\n")
         else:
             f.write(lines)
             
def format_e(n):
    
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'e' + a.split('E')[1]


