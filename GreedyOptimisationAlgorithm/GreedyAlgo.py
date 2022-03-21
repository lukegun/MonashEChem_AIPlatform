#This file has the algorithms used to fit the FTACV data
from copy import deepcopy
import CMAES as standCMA
from scriptformmod import Var2data,format_e
import greedy_mod as gm

class CMAoptsine:

    def __init__(self,varibles,priors,constants,allpara,Yexp):

        self.allpara = allpara      #[label1,label2,...]
        self.varibles = varibles    #[label1,label2,...]
        self.constants = constants #[[label,value]]
        self.priors = priors #[[label,min,max],[label,min,max],...]
        self.Yexp = Yexp

    def sinelooper(self,*args):
        args = args[0]
        #print(args)
        dic = {}
        for items in self.constants:
            dic.update({items[0]:items[1]})



        for i in range(len(self.varibles)):
            #scales the function
            for priorpoints in self.priors:
                if self.varibles[i] == priorpoints[0]:
                    minp = priorpoints[1]
                    maxp = priorpoints[2]
                    diff = maxp-minp
                    break

            value = minp + diff*args[i]

            dic.update({self.varibles[i]: value})

        Ysim = sawtooth(dic,Np=2**10)

        Err = sum((self.Yexp - Ysim)**2)

        return Err

def convergencetest(var, op_settings, header, MEC_set, current,Niter):
    #stores the values from the loop
    rawarray = []
    array = []
    for i in range(var.shape[0]):
        array.append([])
        rawarray.append([])

    meanerror = []
    for i in range(Niter):

        results, logtot = standCMA.STAND_CMAES_TOTCURR(var, op_settings, header[1], MEC_set, current)
        rawfit = results[0]

        fit, Perr = standCMA.CMA_output([rawfit], **MEC_set)
        meanerror.append(Perr) # sores the percentage error

        #array.append(fit)
        for ii in range(var.shape[0]):
            array[ii].append(fit[ii])
            rawarray[ii].append(rawfit[ii])

    return rawarray,array,results,meanerror

def greedypin(var, scaledparas, functparas, funcvar_holder,op_settings, header, MEC_set, current,rawstd,std,boolrawstd, boolRSD,mean):

    # identify worst 3 parameters to test
    spaces = MEC_set.get("spaces")
    stdind,data = stdallocator(rawstd,std, 3,boolrawstd, boolRSD,mean,var,spaces,MEC_set.get("data"))
    MEC_set.update({'data': data})


    print("I need something here to pin the values to the mean")

    layerdic = {}
    Errstore = []
    #constantshold = constants
    scaledparashold = deepcopy(scaledparas)
    functparashold = deepcopy(functparas)
    funcvar_holderhold = deepcopy(funcvar_holder)
    varibleshold = var
    for values in stdind:
        #constants = deepcopy(constantshold) #Constants have been saved to the MECFILE
        var = varibleshold
        scaledparas = deepcopy(scaledparashold)
        functparas = deepcopy(functparashold)
        funcvar_holder = deepcopy(funcvar_holderhold)

        # readjusts the scaled parameters
        scaledparas, functparas, funcvar_holder, var,data,cutdic = listmodifier(values, scaledparas, functparas, funcvar_holder,
                                                                    var,data,mean,spaces)
        print(var)
        print(functparas)
        # update the MEC_set dictionary
        MEC_set.update({'var': var})
        MEC_set.update({'scalvar': scaledparas})
        MEC_set.update({'funcvar': functparas})
        MEC_set.update({'funcvar_holder': funcvar_holder})

        # optimisation loop
        results, logtot = standCMA.STAND_CMAES_TOTCURR(var, op_settings, header[1], MEC_set, current)
        fit = results[0]

        testgreedfit, testgreederror = standCMA.CMA_output([fit], **MEC_set)

        # REMEBER WE ARE MINIMISINE THE ERROR
        """STD WAS REMOVED DUE TO SYSTEM AGRESSIVLY BIASING TO """
        Err = results[1] #- std[values]/10 # considering equations plus the standard deviation of parameters divide by ten

        print("ERROR value: " + str(Err)  +" STD/10: "+str(std[values]/10))

        if len(Errstore) == 0:
            greedvar = values
            rawgreedfit = fit
            greedfit = testgreedfit
            greederror = testgreederror
        elif all(Err <= i for i in Errstore): # this is where the accepting happens (CHANGE above error to something more appropriate
            greedvar = values
            rawgreedfit = fit
            greedfit = testgreedfit
            greederror = testgreederror

        Errstore.append(Err)

        layerdic.update({values: [fit, Err]})

    # sets the greedy point as convergence node
    #constants = deepcopy(constantshold)
    var = deepcopy(varibleshold)
    scaledparas = deepcopy(scaledparashold)
    functparas = deepcopy(functparashold)
    funcvar_holder = deepcopy(funcvar_holderhold)
    #constants.append([greedvar, concdic.get(
        #greedvar)])  # this will need to be replaced by something in the sql or logic in real example

    #premodification for data
    if var.iloc[greedvar][4] >= 40 and var.iloc[greedvar][4] <= 50:
        # this is done in the list modifier function
        pass
    else:
        varslice = var.iloc[greedvar]
        paras = [mean[greedvar], var.iloc[greedvar][4]]
        x = data.iloc[int(var.iloc[greedvar][5])][0]
        data.iloc[int(var.iloc[greedvar][5])] = Var2data(x, paras, spaces)

    # readjusts the scaled parameters
    scaledparas, functparas, funcvar_holder, var,data,cutdic = listmodifier(greedvar,scaledparas,functparas,funcvar_holder,var,data,mean,spaces)


    print(var)
    print(greedvar)
    return rawgreedfit,greedfit,greederror,var,scaledparas,functparas,funcvar_holder,data,cutdic

def stdallocator(rawstd,std, Num,boolrawstd, boolRSD,mean,var,spaces,data):
    x = deepcopy(rawstd)
    y = deepcopy(std)

    logstoredic = {}

    iRSD = 0
    irawstd = 0
    if sum(boolRSD) != 0:
        stdind = [std.index(max(std))]
        logstoredic.update({stdind[-1]:"log"})
        iRSD += 1
        y.pop(stdind[0])
    else:
        stdind = [rawstd.index(max(rawstd))]
        logstoredic.update({stdind[-1]: "notlog"})
        irawstd += 1
        x.pop(stdind[0])

    if Num != 1:
        for i in range(Num - 1):
            if irawstd != sum(boolrawstd):
                stdind.append(rawstd.index(max(x)))
                logstoredic.update({stdind[-1]: "notlog"})
                irawstd += 1
                i = x.index(max(x))
                x.pop(i)
            else: # add
                stdind.append(std.index(max(y)))
                logstoredic.update({stdind[-1]: "log"})
                iRSD += 1
                i = y.index(max(y))
                y.pop(i)


    # sort stdind into reverse order largest to smallest so no cutting issues with var
    stdind.sort()
    stdind = stdind[::-1]

    # THIS FUNCTION WILL BE USED IN THE GREEDY PIN AND MORE GENERAL PIN
    # ISSUE WITH GREEDY PIN THE boolraw std wont work due to all being zero. ADDanother parameter
    # for storage ???
    # [print(keys, dic.get(keys)) for keys in sorted(dic)[::-1]]
    for values in stdind:
        if var.iloc[values][4] >= 40 and var.iloc[values][4] <= 50:
            #this is done in the list modifier function
            pass
        else:
            varslice = var.iloc[values]
            paras = [mean[values], varslice.iloc[4]]
            x = data.iloc[int(varslice.iloc[5])][0]
            data.iloc[int(varslice.iloc[5])] = Var2data(x, paras, spaces)

    """somefunction to pin the FUNC PARAMETERS"""

    return stdind,data

# modifies all the stuff used by the optimiser when the parameter is pinned
def listmodifier(value,scaledparas,functparas,funcvar_holder,var,data,mean,spaces):
    # gets the cut row
    varsslice = var.iloc[value]
    print("cut varible ")
    print(varsslice)
    s = str(varsslice.loc[4])+"-"+str(varsslice.loc[5])
    varsslice.loc[0] = mean[value]
    cutdic = {s:varsslice}

    value = value + 1 # conversion from starts at zero used by computer to sttarts at 1 used by computer

    if scaledparas[0][0] != 0:
        x = deepcopy(scaledparas)
        ii = 1
        Ns = scaledparas[0][0]
        for stuff in scaledparas[1:]:
            if value == stuff[1]:
                # Put the mean value into the data if pinned
                paras = [mean[value-1],stuff[2]]
                ins = Var2data(data.iloc[int(stuff[3])][0],paras,spaces)
                data.iloc[int(stuff[3])][0] = ins
                #pop scaller value in
                x.pop(ii)
                Ns -= 1
                ii -= 1 # due to shortening of x due to above
            elif value < stuff[1]:  # makes the adjustment for the system if the pinned parameter is less then the rest
                x[ii][1] = stuff[1] - 1
            ii += 1

        x[0] = [Ns]
        scaledparas = x

    if functparas[0][0] != 0:
        x = deepcopy(functparas)
        ii = 1
        Ns = functparas[0][0]
        for stuff in functparas[1:]:
            if stuff[2] == 1 and value == stuff[3]:
                x[ii][3] = float(mean[value-1])
                x[ii][2] = 0
                # find the function holder with the corrisponding value and pop it off
                Nf = len(funcvar_holder)
                for j in range(Nf):
                    if value == funcvar_holder[j][6]:
                        funcvar_holder.pop(j)

            elif stuff[2] == 1 and value < stuff[3]:  # same modification for paired case
                x[ii][3] = x[ii][3] - 1



            if value == stuff[1]:  # if pinned value removed remover from stuff
                if stuff[2] == 1:  # same modification for paired case
                    # if paired swap the values
                    y = [x[ii][0],x[ii][3],0,mean[value-1]]
                    x[ii] = y
                else:
                    #l = functionallist[j][0]
                    #bunch of modification to inputs so I don't have to rewrite a function

                    """NEED SOMETHING HERE TO FIND WHICH FUNCvar corrispponds to the thing"""
                    testfuncpara = [[1],x[ii]]
                    print(x)
                    print(x[ii][1])
                    for hell in funcvar_holder:
                        if hell[-1] == x[ii][1]: #checks to see if the functional para is the same
                            print(hell)
                            functionallist = [[mean[int(x[ii][1])-1],int(hell[-3]),int(hell[-2])]]
                            break

                    #calculate kb and kf
                    pinnedfunctionparas = gm.listin_func(mean, testfuncpara, functionallist)

                    #assume only one at time
                    if pinnedfunctionparas[0][2] == 1:  # Keq
                        # setting Kf fisrt one
                        x = data.iloc[int(pinnedfunctionparas[0][3])][0].split(',')  # breaks array input into sections
                        x[spaces[3] + 1] = format_e(pinnedfunctionparas[0][0])  # inserts value at 1 point after reaction mechanisim
                        x[spaces[3] + 2] = format_e(pinnedfunctionparas[0][1]) # set kb
                        data.iloc[int(pinnedfunctionparas[0][3])][0] = ",".join(x)  # rejoins x into a script

                    else:
                        print('wrong inputs yoo')

                    # Now we can delete the function
                    x.pop(ii)
                    Ns -= 1

                funcvar_holder.pop(ii-1)  # pops off the function holder

            elif value < stuff[1]:  # modification for if scalarvalue is greater then removed para
                x[ii][1] = x[ii][1] - 1
            ii += 1

        x[0] = [Ns]
        functparas = x

    # modification for func holder in case that pararemoved is less then  th original para
    if len(funcvar_holder) != 0:  # case where its empty
        for ii in range(len(funcvar_holder)):
            if value < funcvar_holder[ii][6]:
                funcvar_holder[ii][6] = funcvar_holder[ii][6] - 1

    # constants.append([values, concdic.get(            values)])  # this will need to be replaced by something in the sql or logic in real example
    var = var.drop(var.index[value-1]) #correction for above standardisation

    return scaledparas,functparas,funcvar_holder, var, data,cutdic

def itterationprinter(filename, iteration,var,mean,std,rawstd,array,noshift):

    f = open(filename,"w+")
    f.write("Iteration " + str(iteration)+"\n\n")

    f.write("current varible optimistation\n")
    for i in range(var.shape[0]):
        for j in range(var.shape[1]):
            f.write(format_e(var.iloc[i][j])+"\t")
        f.write("\n")

    f.write("\nOptimised mean \u00B1 std\n")
    for i in range(len(mean)):
        f.write(format_e(mean[i])+" \u00B1 " +format_e(std[i]*mean[i])+ "\n")

    f.write("\noptimised values\n")
    for stuff in array:
        for values in stuff:
            f.write(format_e(values)+ "\t")
        f.write("\n")

    f.write("\nraw std\n")
    for values in rawstd:
        f.write(format_e(values)+"\t")
    f.write("\n")

    f.write("\nRSD\n")
    for values in std:
        f.write(format_e(values) + "\t")
    f.write("\n")

    f.write("\npriors that didn't shift\n")
    for values in noshift:
        f.write(str(int(values)) + "\t")
    f.write("\n")


    f.close()

    return

#sets up a header file for the output of the POT compadable file
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
    for i in range(1):
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


# function to get around python printing numbers truncated and correct format
def format_e_out(n):

    a = '%E' % n
    v = a.split('E')[0].rstrip('0')
    for i in range(len(v),8):
        v += '0'
    v +='E' + a.split('E')[1]

    return  v

# writes the MECsim output to a txt file compadable with POT
def outputwriter(filename,startpart,voltage,Scurr,timev):

    name = filename + '.txt'

    f = open(name, 'w')
    f.write(startpart)

    for i in range(len(Scurr)):
        s = format_e_out(voltage[i]) + '   ' + format_e_out(Scurr[i]) + '   ' \
            + format_e_out(timev[i]) + '\n'
        f.write(s)

    return

#converts the row line back to repeat
def varconverter(var,scaledparas,funcvar,spaces):
    Np = var.shape[0]

    for i in range(Np):
        code = var.iloc[i][4]
        row = var.iloc[i][5]

        if int(code/10) == 2:
            x = row - spaces[1] + 1
        elif int(code/10) == 3:
            x = row - spaces[2] + 1
        elif int(code/10) == 4:
            x = row - spaces[2] + 1
        else:
            x = row

        var.loc[var.index[i], 5] = x

    for i in range(1,len(scaledparas)):
        code = scaledparas[i][2]
        row = scaledparas[i][3]

        if int(code / 10) == 2:
            x = row - spaces[1] + 1
        elif int(code / 10) == 3:
            x = row - spaces[2] + 1
        elif int(code / 10) == 4:
            x = row - spaces[2] + 1
        else:
            x = row

        scaledparas[i][3] = x

    return var,scaledparas,funcvar

def bayespriorshift(var,meanhold):
    # puts the mean value in var
    for i in range(len(meanhold)):
        mean = meanhold[i]
        var.loc[var.index[i], 1] = mean
        code = var.iloc[i][4]

        if code == 11: # Resistance
            if mean < 25:
                low = 1
                high = 50
            elif mean < 101:
                low = 1
                high = 100
            else:
                low = mean - 100
                high = mean + 100

        elif code == 22: # Diffusion
            shift = mean*0.3
            low = mean - shift
            high = mean + shift

        elif code == 31 or code == 32:    # kb and kf
            shift = mean * 0.3
            low = mean - shift
            high = mean + shift

        elif code == 33:
            high = mean + 0.010
            low = mean - 0.010
        elif code == 34:
            low = mean/10
            high = mean*3

        elif code == 35:
            low = mean - 0.1
            high = mean + 0.1
        elif code == 41:
            low = mean / 10
            high = mean * 3

        elif code == 42:
            if mean < 79.99:
                high = 89.9999
                low = 2*mean - high
            else:
                high = mean + 10
                low = mean - 10


        # places where this goes
        var.loc[var.index[i], 0] = mean
        var.loc[var.index[i], 1] = low
        var.loc[var.index[i], 2] = high

    return var