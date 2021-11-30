# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:52:59 2018

@author: Luke Gundry
"""

from pandas import read_csv
import numpy as np
from subprocess import run


#writes MECSim
def MECSimwriter(filename,data):
    
    n = len(data.index)
    i = 0 
    f = open(filename,"w") # opens the input file for MECSIM at start ready for it to be rewritten
    while i < n: #tells the competer to go through everyline
    
        f.write(str(data.iloc[i][0]))
        f.write("\n")   # new line on txt
        i += 1

    f.close()

# reads the initial input file
def globalinreader(filename):
    
    Input_file = filename #input("Name of the textfile input:") # will need to be modified to be used with the 
    globalin = read_csv(Input_file,sep='    ',index_col=False,header = None,comment='!')
    
    nCMA_settings = globalin[globalin[0].str.match('MECSim settings')].index.values # gets the CMA settings seperator
    nExp = globalin[globalin[0].str.match('Experimental settings')].index.values      
    
    #splits the inpfile into its important sections
    CMA_settings = globalin[0:nCMA_settings[0]]
    data = globalin[nCMA_settings[0] + 1:nExp[0]] # will need to be fixed in future updates to truncate at exdata
    Exp_data = globalin[nExp[0] + 1::]  # experimental data
    
    #This needs to be moved till after CMA-Settings
    #Exp_data = Exp_data[0].str.split('  ',expand=True).astype('float')# \t for tab sep files # '  ' for mecsim
    
    return CMA_settings, data, Exp_data

# splits exp data depending on data type
def Exp_data_spliter(Exp_data, datatype):
    
    if datatype == 0: # MECsim simulation data (of form {v,i,t})
        Exp_data = Exp_data[0].str.split('  ',expand=True).astype('float')
        
        #sets up for usable information
        curr = np.array(Exp_data.values)
        Exp_t = curr[1,2]
        curr = curr[:,1]

        
    elif datatype == 1:  # FTACV experimental data (of form {v,i,t})
        Exp_data = Exp_data[0].str.split('\t',expand=True).astype('float')
        
        #sets up for usable information
        curr = np.array(Exp_data.values)
        Exp_t = curr[1,2]
        curr = curr[:,1]
        
    elif datatype == 2: # CHI data type (of form {v,i}) (NEED SMETHING TO get time)
        Exp_data = Exp_data[0].str.split(',',expand=True).astype('float')
        
        #sets up for usable information
        curr = np.array(Exp_data.values)
        curr = curr[:,1]
        Exp_t = 'none' # (CHI IS DC ONLY SO TIME DOMAIN DOEN't matter)
        
    else:
        print('need a data type to compair to')
        
    return curr, Exp_t

# extracts the CMA settings from the block data
def CMA_Varibles(CMA_settings):
    
    nvar = int(CMA_settings.iloc[0][0])
    n = CMA_settings[0].shape
    n = int(n[0])
    nscal = int(CMA_settings.iloc[nvar + 1][0])      # gets number of scaling parameters
    
    # Gets the scaling 
    scalvarpre = CMA_settings[nvar+1:nvar+2+nscal] # gets the num of scaling to
    # turns scalvarpre into a list of values
    scalvar = [[int(scalvarpre.iloc[0][0])]]
    i = 0 
    x = scalvarpre[0].str.split(',',expand=True).astype('float')
    x = x.values
    
    while i != scalvar[0][0]:
        scalvar.append(x[i + 1,:])
        i += 1    
    
    Nlist = 6
    nband = int((n - Nlist - nvar - 1 - nscal)/2)
    
    #Seperates the CMA input data
    var = CMA_settings[1:nvar + 1]
    bandwidth = CMA_settings[nvar+ 1+ 1 + nscal:nvar+ 1 +nband + 1 + nscal]
    harm_weights = CMA_settings[nvar+ 1 +nband + 1 + nscal:nvar+ 1 + 2*nband + 1 + nscal]
    
    #Changes the seperated input values into 
    var = var[0].str.split(',',expand=True).astype('float')
    bandwidth = bandwidth[0].str.split(',',expand=True).astype('float')
    bandwidth = bandwidth.values    # changes from a df to np.array
    harm_weights = harm_weights[0].str.split(',',expand=True).astype('float')
    harm_weights = harm_weights.values
    
    
    #Optimization settings
    Np = nvar + 2*nband + 1 + nscal     # starting point
    op_settings = [0,0,0,0,0]     #prealocation
    op_settings[0] = int(CMA_settings.iloc[Np + 1][0])   # This one setts the optimization method
    op_settings[1] = int(CMA_settings.iloc[Np + 3][0]) # this one does the num cores
    datatype = int(CMA_settings.iloc[Np + 2][0]) #Gets the type of data were compairing against
    op_settings[2] = 2**float(CMA_settings.iloc[Np + 4][0])  # sets the number of datapoints to compair in the current
    op_settings[3] = float(CMA_settings.iloc[Np + 5][0])  # sets the tolx value
    op_settings[4] = float(CMA_settings.iloc[Np + 6][0])  # sets the initial sigma
    
    return var, bandwidth, harm_weights, op_settings, datatype, scalvar # Modified CMA VALUES var

#extracts the scanspace settings from the block data
def preMCMC_Varibles(settings):
    
    nvar = int(settings.iloc[0][0])
    n = settings[0].shape
    n = int(n[0])
    
    nband = int((n-4-nvar)/2)  #gets the number of bandwidths
    
    #Seperates the CMA input data
    var = settings[1:nvar + 1]
    bandwidth = settings[nvar+1:nvar+ 1 +nband]
    harm_weights = settings[nvar+ 1 +nband:nvar+ 1 + 2*nband]
    
    #Changes the seperated input values into 
    var = var[0].str.split(',',expand=True).astype('float')
    bandwidth = bandwidth[0].str.split(',',expand=True).astype('float')
    bandwidth = bandwidth.values    # changes from a df to np.array
    harm_weights = harm_weights[0].str.split(',',expand=True).astype('float')
    harm_weights = harm_weights.values
    
    #Optimization settings
    op_settings = [0,0,0]     #prealocation
    op_settings[0] = int(settings.iloc[nvar+ 1 + 2*nband][0])   # This one setts the optimization method
    #op_settings[1] = int(CMA_settings.iloc[nvar+ 1 + 2*nband + 1][0]) # this one does the num cores
    datatype = int(settings.iloc[nvar+ 2 + 2*nband][0]) #Gets the type of data were compairing against
    res = int(settings.iloc[nvar+ 3 + 2*nband][0]) 
    op_settings[2] = 2**float(settings.iloc[nvar+ 4 + 2*nband][0])  # sets the number of datapoints to compair in the current
    
    return var, res ,bandwidth, harm_weights, op_settings, datatype # Modified CMA VALUES var

# take the harmonic weight input values and changes them to a vector
def harm_weights_trans(harm_weights):
    
    ###weights need to go through a reshape row vector
    fund_harm_wieght = np.array(harm_weights[0,0])
    x = harm_weights[1::,:]             # collects harmonic wieghts
    nx = np.prod(x.size)                # gets the number of elements of x
    x = np.squeeze(np.asarray(x.reshape(1,nx))) # turns the matrix into an array
    harm_weights = np.append(fund_harm_wieght,x)

    return harm_weights

# this reads MECSim settings and tells the copmuter what line everything is on
def data_finder(data):

    i =22  # dataframe index always same
    j = 0
    # checks to see if number of cycles is meet
    while int(float(data.iloc[i][0])) != float(data.iloc[i][0]) :
        i += 1
        j += 1
        
    Crot = i + 10 # add values to next varible rotation speed
    Nac = int(data.iloc[Crot + 1][0]) # Number of AC inputs
    Nsol =  int(data.iloc[Crot + 2 + Nac][0]) # number of sols pres
    Csol = Crot + 2 + Nac + 1 # Line solution propities start n 
    Ncap = int(data.iloc[Csol+ Nsol][0])
    Ckin = Csol+ Nsol + Ncap + 3
    
    spaces = [Crot, Csol, Ckin,Nsol,Nac]
    
    return spaces

def freq_test(var):

    i = 0
    freqvar = [False,0]
    while i != len(var.index):
        # checks for frequency as a varible
        if var.iloc[i][4] == 8:
            freqvar = [True,i]
        i += 1

    return freqvar


# Assigns the varibles to there respective lines in the MECSim settings
def line_assigner(var, spaces, scalvar):
    # sets the value for var
    print(spaces)
    i = 0
    while i != len(var.index):

        # Misc varibles
        if var.iloc[i][4] == 11:  # resistance
            var.loc[i,5] = 1

        elif var.iloc[i][4] == 1: # temp
            var.loc[i,5] = 0

        elif var.iloc[i][4] == 2:  # Estart
            var.loc[i,5] = 2
        elif var.iloc[i][4] == 3: # Erev
            var.loc[i,5] = 3
        elif var.iloc[i][4] == 4: # scanrate
            var.loc[i,5] = 5
        elif var.iloc[i][4] == 6: # Surface Area
            var.loc[i,5] = spaces[0] - 9
        elif var.iloc[i][4] == 7 or  var.iloc[i][4] == 8 : # amp and frequency
            var.loc[i,5] = spaces[0] + var.iloc[i][5] +1
        elif var.iloc[i][4] == 12:  # kinematic vis
            var.loc[i,5] = spaces[0]

        elif var.iloc[i][4] == 13:  # NOISE PARAMETER
            pass

        elif var.iloc[i][4] > 20 and var.iloc[i][4] < 30:  # Solution block propities
            var.loc[i, 5] = spaces[1] + var.iloc[i][5] - 1

        elif var.iloc[i][4] > 30 and var.iloc[i][4] < 40:  # Kinetic propities
            var.loc[i,5] = spaces[2] + var.iloc[i][5] - 1

        elif var.loc[i][4] == 41 or var.iloc[i][4] == 42:  # functional propities for Keq (41= r, 42 = theta)
            var.loc[i, 5] = spaces[2] + var.iloc[i][5] - 1

        elif var.loc[i][4] == 43:  # E0 when the range is defined by two other varibles
            var.loc[i, 5] = spaces[2] + var.iloc[i][5] - 1
        elif var.loc[i][4] == 44:  # varing a function as a scalar of another varible allot this at a seperate section
            pass

        elif var.iloc[i][4] > 50 and var.iloc[i][4] < 60:  # capacitance propities for Keq (41= r, 42 = theta)
            var.loc[i, 5] = spaces[2] - 6 + var.iloc[i][4] - 50
            # 51 = C0, 52 = C1, 53 = C2, 54 = C3, 55 = C4, 56 = Scalar*(C)


        else:
            print('Error in script_generator function line_assigner')

        i += 1

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

    return var, scalvar

# function to get around python printing numbers truncated and worng0
def format_e(n):
    
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'e' + a.split('E')[1]

# draft data handeler to put varibles into data df for printing a MECSim input 
def Var2data(data,var,val_in,spaces):    
    
    n = len(val_in)

    i = 0   
    while i != n:  # this whole section could be made more efficent but I don't know.

        if (int(var.iloc[i][4]) != 7 and int(var.iloc[i][4]) != 8) and var.iloc[i][4] < 20 : # assigns all Misc varibles
            data.iloc[int(var.iloc[i][5])][0] = format_e(val_in[i])

        elif int(var.iloc[i][4]) == 7: # sine amp
            x = data.iloc[int(var.iloc[i][5])][0].split(',')  # breaks array input into sections
            data.iloc[int(var.iloc[i][5])][0] = '%s, %s' % (format_e(val_in[i]),x[1])

        elif int(var.iloc[i][4]) == 8: # sine freq
            x = data.iloc[int(var.iloc[i][5])][0].split(',')  # breaks array input into sections
            data.iloc[int(var.iloc[i][5])][0] = '%s, %s' % (x[0],format_e(val_in[i]))

        elif var.iloc[i][4] == 21: # concentration
            x = data.iloc[int(var.iloc[i][5])][0].split(',') # breaks array input into sections  
            data.iloc[int(var.iloc[i][5])][0] = '%s,%s,%s' % (format_e(val_in[i]), x[1], x[2])
            
        elif var.iloc[i][4] == 22: # diffusion
            x = data.iloc[int(var.iloc[i][5])][0].split(',') # breaks array input into sections
            data.iloc[int(var.iloc[i][5])][0] = '%s, %s,%s' % (x[0], format_e(val_in[i]), x[2] )
            
        elif var.iloc[i][4] == 31: # reaction rate forward
            x = data.iloc[int(var.iloc[i][5])][0].split(',') # breaks array input into sections
            x[spaces[3]+1] = format_e(val_in[i])  # inserts value at 1 point after reaction mechanisim
            data.iloc[int(var.iloc[i][5])][0] = ",".join(x)     # rejoins x into a script
            
        elif var.iloc[i][4] == 32: # reaction rate backward
            x = data.iloc[int(var.iloc[i][5])][0].split(',') # breaks array input into sections
            x[spaces[3]+2] = format_e(val_in[i])
            data.iloc[int(var.iloc[i][5])][0] = ",".join(x)
            
        elif var.iloc[i][4] == 33: # Reaction potential Eo
            x = data.iloc[int(var.iloc[i][5])][0].split(',') # breaks array input into sections
            x[spaces[3]+3] = format_e(val_in[i])
            data.iloc[int(var.iloc[i][5])][0] = ",".join(x)
            
        elif var.iloc[i][4] == 34: # electon kinetic reaction rate
            x = data.iloc[int(var.iloc[i][5])][0].split(',') # breaks array input into sections
            x[spaces[3]+4] = format_e(val_in[i])
            data.iloc[int(var.iloc[i][5])][0] = ",".join(x)
            
        elif var.iloc[i][4] == 35: # alp or lambda
            x = data.iloc[int(var.iloc[i][5])][0].split(',') # breaks array input into sections
            x[spaces[3]+5] = format_e(val_in[i])
            data.iloc[int(var.iloc[i][5])][0] = ",".join(x)

        elif var.iloc[i][4] == 41 or var.iloc[i][4] == 42 or var.iloc[i][4] == 43 or var.iloc[i][4] == 44 :  # functional parameters
            pass
            
        elif 50 < var.iloc[i][4] < 60 :
            pass

        else:
            print('error in allocation module Var2Data')
        
        i += 1
    
    return data


# writes the functional varibles to a data file
def funcvar2data(funclist, spaces, data):
    # loop designed so that other functionals can be added
    N = len(funclist)
    i = 0
    while i != N:
        if funclist[i][2] == 1:  # Keq
            # setting Kf fisrt one
            x = data.iloc[int(funclist[i][3])][0].split(',')  # breaks array input into sections
            x[spaces[3] + 1] = format_e(funclist[i][0])  # inserts value at 1 point after reaction mechanisim
            data.iloc[int(funclist[i][3])][0] = ",".join(x)  # rejoins x into a script

            # setting Kb
            x = data.iloc[int(funclist[i][3])][0].split(',')  # breaks array input into sections
            x[spaces[3] + 2] = format_e(funclist[i][1])
            data.iloc[int(funclist[i][3])][0] = ",".join(x)

        elif funclist[i][2] == 2: # formal potnetial value
            x = data.iloc[int(funclist[i][3])][0].split(',')  # breaks array input into sections
            x[spaces[3] + 3] = format_e(funclist[i][0])
            data.iloc[int(funclist[i][3])][0] = ",".join(x)

        elif funclist[i][2] == 3: # formal potnetial value
            x = data.iloc[int(funclist[i][3])][0].split(',')  # breaks array input into sections
            x[spaces[3] + 4] = format_e(funclist[i][0])
            data.iloc[int(funclist[i][3])][0] = ",".join(x)

        else:
            print('wrong inputs yoo')
        i += 1

    return data

#applies the scaling for
def cap2data(val_in, var, spaces, data, cap_series):
    # loop designed so that other functionals can be added
    N = len(val_in)

    i = 0
    while i != N:
        if var.iloc[i][4] == 51: # C0
            #x[spaces[3] + 5] = format_e(val_in[i])
            data.iloc[int(var.iloc[i][5])][0] = format_e(val_in[i])
        elif var.iloc[i][4] == 52:  # C1
            #x[spaces[3] + 5] = format_e(val_in[i])
            data.iloc[int(var.iloc[i][5])][0] = format_e(val_in[i])
        elif var.iloc[i][4] == 53:  # C2
            #x[spaces[3] + 5] = format_e(val_in[i])
            data.iloc[int(var.iloc[i][5])][0] = format_e(val_in[i])
        elif var.iloc[i][4] == 54:  # C3
            #x[spaces[3] + 5] = format_e(val_in[i])
            data.iloc[int(var.iloc[i][5])][0] =  format_e(val_in[i])
        elif var.iloc[i][4] == 55:  # C4
            #x[spaces[3] + 5] = format_e(val_in[i])
            data.iloc[int(var.iloc[i][5])][0] = format_e(val_in[i])
        elif var.iloc[i][4] == 56:  # Scal*(c)

            Csol = int(data.iloc[spaces[1] - 1][0])
            j = 0
            while j !=len(cap_series):
                data.iloc[spaces[1]+Csol +2+j][0] = format_e(val_in[i]*cap_series[j])
                j += 1
        else:
            pass
        i += 1


    return data

# inserts the scalar dependant varibles into data file
def Scalvar2data(scalarvar, val_in, data, spaces):
    
    n = scalarvar[0][0] # gets number of scalar varibles
    
    i = 1 
    while i != n+1:  # this whole section could be made more efficent but I don't know
        
        scal = scalarvar[i][0]*val_in[int(scalarvar[i][1] - 1)] # scales the varible
        
        if scalarvar[i][2] < 20: # assigns all Misc varibles
            data.iloc[int(scalarvar[i][3])][0] = format_e(scal)
            
        elif scalarvar[i][2] == 21: # concentration
            x = data.iloc[int(scalarvar[i][3])][0].split(',') # breaks array input into sections  
            data.iloc[int(scalarvar[i][3])][0] = '%s,%s,%s' % (format_e(scal), x[1], x[2])
            
        elif scalarvar[i][2] == 22: # diffusion
            x = data.iloc[int(scalarvar[i][3])][0].split(',') # breaks array input into sections
            data.iloc[int(scalarvar[i][3])][0] = '%s, %s,%s' % (x[0], format_e(scal), x[2] )
            
        elif scalarvar[i][2] == 31: # reaction rate forward
            x = data.iloc[int(scalarvar[i][3])][0].split(',') # breaks array input into sections
            x[spaces[3]+1] = format_e(scal)  # inserts value at 1 point after reaction mechanisim
            data.iloc[int(scalarvar[i][3])][0] = ",".join(x)     # rejoins x into a script
            
        elif scalarvar[i][2] == 32: # reaction rate backward
            x = data.iloc[int(scalarvar[i][3])][0].split(',') # breaks array input into sections
            x[spaces[3]+2] = format_e(scal)
            data.iloc[int(scalarvar[i][3])][0] = ",".join(x)
            
        elif scalarvar[i][2] == 33: # Reaction potential Eo
            x = data.iloc[int(scalarvar[i][3])][0].split(',') # breaks array input into sections
            x[spaces[3]+3] = format_e(scal)
            data.iloc[int(scalarvar[i][3])][0] = ",".join(x)
            
        elif scalarvar[i][2] == 34: # electon kinetic reaction rate
            x = data.iloc[int(scalarvar[i][3])][0].split(',') # breaks array input into sections
            x[spaces[3]+4] = format_e(scal)
            data.iloc[int(scalarvar[i][3])][0] = ",".join(x)
            
        elif scalarvar[i][2] == 35: # alp or lambda
            x = data.iloc[int(scalarvar[i][3])][0].split(',') # breaks array input into sections
            x[spaces[3]+5] = format_e(scal)
            data.iloc[int(scalarvar[i][3])][0] = ",".join(x)

        elif 60 >scalarvar[i][2] > 50:
            pass

        else:
            print('error in allocation module Scal2Data')
        
        i += 1
    
    return data

# writes data.df into a .inp for MECSim to run
def Mec_writer(data):
    
    n = len(data.index) #samples the length of the datafile
    i = 0 
    f = open("Master.inp","w")    # creates a blank .inp to be written
    while i < n:
        f.write(str(data.iloc[i][0]))
        f.write("\n")  # Getting a new line
        i += 1
    f.close()
    
# reads mecsim output
def output_reader():
    
    results  = read_csv('MECSimOutput_Pot.txt', delim_whitespace = True, skiprows=41,names ={"v", "i","t"})
    results = results.values    # changes results fro df object to np array
    
    return results

"""DC and AC ethods could be conbined to one printer that takes in DCACethod to make mild changes"""
def CMA_output_printDC(t2,space_holder, var_out,Perr,res,c,mean_var_out):
    # t2,space_holder, var_out,Perr,res,c
    # For AC the main difference is that Perr is a matrix 
    
    filename = "global_out.txt"
    f = open(filename,"w")    # creates a blank .inp to be written
    
    #writes the input 
    f.write('Input file name: %s\n' %(filename))
    
    #inserts time
    f.write('Completetion time (min): %f\n\n' %(t2))
   
    
    # var_out
    f.write('Optimized paramters;{val,code,repeat}\n')
    N = len(var_out)
    x = [0,0,0]
    i = 0
    while i != N:
        x[0] = var_out[i]   #var out is a string
        x[1] = str(int(space_holder[i,0]))
        x[2] = str(int(space_holder[i,1]))
        y = ",".join(x)
        f.write('%s\n' %(y))
        i += 1
    f.write('\n')
    
    # Perr different to AC
    x = format_e(Perr)
    f.write('Percentage Error: %s\n\n' %(x))
    
    # res
    f.write('Misc CMA_ES output\n')
    x = res[1]
    x = format_e(x)
    f.write('Difference squared: %s\n' %(x))
    
    # mean_var_out = str(mean_var_out)
    f.write('CMA-ES Mean parametervalues (mean,std):\n')
    i = 0
    while i != N:
        
        # formats values into something we can use
        x1 = mean_var_out[0][i]
        x2 = mean_var_out[1][i]
        m = format_e(x1)
        s = format_e(x2)
        
        f.write('Var %d: %s,%s\n' %(int(i+1),m,s))
        i +=1
        
    f.write('\n')
    
    #theres the other output values just need to know what they are 
    f.write('CMA-ES number of function evaluations: %s\n' %(res[3]))
    # CMAstd = res[6] STD cannot be done due to the assymetrical propities of the transform
    
    f.close()
    
    return

def CMA_output_printAC(t2,space_holder, var_out,Perr,res,c,mean_var_out):
    
    # For AC the main difference is that Perr is a matrix 
    
    filename = "global_out.txt"
    f = open(filename,"w")    # creates a blank .inp to be written
    
    #writes the input 
    f.write('Input file name: %s\n' %(filename))
    
    #inserts time
    f.write('Completetion time: %f\n\n' %(t2))
   
    
    # var_out
    f.write('Optimized paramters;{val,code,repeat}\n')
    N = len(var_out)
    x = [0,0,0]
    i = 0
    while i != N:
        x[0] = str(format_e(var_out[i]))   #var out is a string
        x[1] = str(int(space_holder[i,0]))
        x[2] = str(int(space_holder[i,1]))
        y = ",".join(x)
        f.write('%s\n' %(y))
        i += 1
    f.write('\n')
    
    # Perr different to AC
    f.write('Percentage Error Values for each harmonic (lowest-highest freq):\n')
    
    H = np.count_nonzero(Perr)   #Gets the number of fittered haarmonics
    Sum_err = str(np.sum(Perr)/H)   #Retreaves Sum error in all harmonics
    x = np.array_str(Perr)  #Converts Perr to a list 
    f.write(x)       #Might need to be fixed
    f.write('\n')    #seperator
    f.write('Total prcentage error of all harmonics: %s\n\n' %(Sum_err))
    
    # res
    f.write('Misc CMA_ES output\n')
    x = res[1]
    x = format_e(x)
    f.write('Difference squared: %s\n' %(x))
  
    # mean_var_out = str(mean_var_out)
    f.write('CMA-ES Mean parametervalues (mean,std):\n')
    i = 0
    while i != N:
        
        # formats values into something we can use
        x1 = mean_var_out[0][i]
        x2 = mean_var_out[1][i]
        m = format_e(x1)
        s = format_e(x2)
        
        f.write('Var %d: %s,%s\n' %(int(i+1),m,s))
        i +=1
        
    f.write('\n')
    #theres the other output values just need to know what they are 
    f.write('CMA-ES number of function evaluations: %s\n' %(res[3]))
    
    f.close()
    
    s = 'done'
    
    return s
 
def logger(s,val_in):
    
    recordfilename = s
    f = open(recordfilename,"w")
    i = 0
    while i!= len(val_in):
        f.write('Var%i:\t' %(i+1))
        i += 1
    f.write('fit_val\n')
    f.close()
    
    return

#Multiprocessing logger call
def iter_logger(filename,val_in):

    f = open(filename,"w+") # nasty but it works
    jj = 0
    Nvar = len(val_in[0]) - 1

    i = 0
    while i != Nvar:
        f.write('Var%i:\t' %(i+1))
        i += 1
    f.write('fit_val\n')

    for num in val_in:
       
        i = 0
        while i != Nvar:
            x =  num[i]
            
            f.write('%s\t' %(format_e(x)))
            i += 1
        f.write(format_e(num[i]))
        f.write('\n')
        jj += 1
    #f.write('%f\n' %(c))
    f.close()
    
    return
