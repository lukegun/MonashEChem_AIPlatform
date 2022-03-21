# a range of functions that handle inputs and outputs and general formatting  of stuff
from pandas import read_csv

# loads a generic masterfile from a default master file
def mastergenericloader(reactmech):
    fileloc = "genericmasterfiles/"

    # this is here incase there is an error loading the generic file
    try:
        data = read_csv(fileloc+ reactmech + ".txt",sep='    ',index_col=False,header = None,comment='!')
    except:
        print("ERROR generic master file not found")
        exit()
    finally:
        pass

    return data

# edits the master file and makes the changes to allow accurate modelling and gives the pinned constants
def values2Master(data,Evalues, ACmode,ACsettings, scanrate,Nsim,pinnedvalues,spaces,capparas):

    # inserts the pinned values
    for values in pinnedvalues:
        x = data.iloc[values[2]][0]
        x = Var2data(x, values, spaces)
        data.iloc[values[2]][0] = x

    # adds the capacitance values in
    for i in range(len(capparas)-1):
        Csol = int(data.iloc[spaces[1] - 1][0])
        data.iloc[spaces[1] + Csol + 2 + i][0] = format_e(capparas[i])

    # Fix the scanrate
    data.iloc[5][0] = format_e(scanrate/1000) # mV/s

    # fix the Nsim
    data.iloc[6][0] = str(int(Nsim))

    #fix the Ncycle number
    data.iloc[4][0] = str(int(Evalues[-1]))

    # Fix the potential values Evalues = [Estart, Eend, Emax, Emin, Ncycle]
    print(Evalues)
    if Evalues[0] == Evalues[1] and Evalues[0] == Evalues[2]:
        reaction = [-1, 1]
        data.iloc[2][0] = format_e(Evalues[0]/1000) #Estart
        data.iloc[3][0] = format_e(Evalues[3]/1000) #Eswitching
    elif Evalues[0] == Evalues[1] and Evalues[0] == Evalues[3]:
        reaction = [1,-1]
        data.iloc[2][0] = format_e(Evalues[0]/1000)  # Estart
        data.iloc[3][0] = format_e(Evalues[2]/1000)  # Eswitching
    else:
        print("ERROR: the evalues aren't in cycle format and hasn't been set up yet")
        exit()

    # Fix the oxidation (Will be different for each reaction mechanism)
    for j in range(spaces[2],len(data.iloc[:])):
        if int(data.iloc[j][0].split(",")[0]) == 0: # checks for a electron transfer and if its there change the values
            nline = j
            newreact = []
            for i in range(nline-spaces[2]):
                newreact.append(str(0))
            newreact.append(str(reaction[0]))
            newreact.append(str(reaction[1]))

            for i in range(Csol -(nline-spaces[2]+ 2 )):
                print("cuntfuck")
                newreact.append(str(0))

            x = data.iloc[j][0].split(",")

            s = x[0] + ", "

            for stuff in newreact:
                s += stuff + ", "

            #adds the rest of the stuff
            for stuff in x[Csol+1:-1]:
                s += stuff + ", "

            s += x[-1] # adds alpha or lambda

            data.iloc[j][0] = s

    # set the AC amp and frequency
    if ACmode:
        s = "%s, %s" %(ACsettings[1],ACsettings[0])
    else:
        s = "0, 9.02"
    data.iloc[spaces[0] + 2][0] = s

    # set the BV MH VALUE (Not required but the option is here)

    return data

# function to get around python printing numbers truncated and worng0
def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'e' + a.split('E')[1]


# draft data handeler to put varibles into data df for printing a MECSim input
def Var2data(x, paras,spaces):

    if (int(paras[1]) != 7 and int(paras[1]) != 8) and paras[1] < 20:  # assigns all Misc varibles
        x = format_e(paras[0])

    elif int(paras[1]) == 7:  # sine amp
        x = x.split(',')  # breaks array input into sections
        x = '%s, %s' % (format_e(paras[0]), x[1])

    elif int(paras[1]) == 8:  # sine freq
        x = x.split(',')  # breaks array input into sections
        x = '%s, %s' % (x[0], format_e(paras[0]))

    elif paras[1] == 21:  # concentration
        x = x.split(',')  # breaks array input into sections
        x = '%s,%s,%s' % (format_e(paras[0]), x[1], x[2])

    elif paras[1] == 22:  # diffusion
        x = x.split(',')  # breaks array input into sections
        x = '%s, %s,%s' % (x[0], format_e(paras[0]), x[2])

    elif paras[1] == 31:  # reaction rate forward
        x = x.split(',')  # breaks array input into sections
        x[spaces[3] + 1] = format_e(paras[0])  # inserts value at 1 point after reaction mechanisim
        x = ",".join(x)  # rejoins x into a script

    elif paras[1] == 32:  # reaction rate backward
        x = x.split(',')  # breaks array input into sections
        x[spaces[3] + 2] = format_e(paras[0])
        x = ",".join(x)

    elif paras[1] == 33:  # Reaction potential Eo
        x = x.split(',')  # breaks array input into sections
        x[spaces[3] + 3] = format_e(paras[0])
        x = ",".join(x)

    elif paras[1] == 34:  # electon kinetic reaction rate
        x = x.split(',')  # breaks array input into sections
        x[spaces[3] + 4] = format_e(paras[0])
        x = ",".join(x)

    elif paras[1] == 35:  # alp or lambda
        x = x.split(',')  # breaks array input into sections
        x[spaces[3] + 5] = format_e(paras[0])
        x = ",".join(x)

    elif paras[1] == 41 or paras[1] == 42 or paras[1] == 43 or paras[1] == 44:  # functional parameters
        pass

    elif 50 < paras[1] < 60:
        pass

    else:
        print('error in allocation module Var2Data')


    return x
