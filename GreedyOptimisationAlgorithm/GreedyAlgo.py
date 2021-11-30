#This file has the algorithms used to fit the FTACV data
from copy import deepcopy
import CMAES as standCMA
from scriptformmod import Var2data
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
        scaledparas, functparas, funcvar_holder, var,data = listmodifier(values, scaledparas, functparas, funcvar_holder,
                                                                    var,data,mean)

        # update the MEC_set dictionary
        MEC_set.update({'var': var})
        MEC_set.update({'scalvar': scaledparas})
        MEC_set.update({'funcvar': functparas})
        MEC_set.update({'funcvar_holder': funcvar_holder})

        # optimisation loop
        results, logtot = standCMA.STAND_CMAES_TOTCURR(var, op_settings, header[1], MEC_set, current)
        fit = results[0]

        # REMEBER WE ARE MINIMISINE THE ERROR
        """STD WAS REMOVED DUE TO SYSTEM AGRESSIVLY BIASING TO """
        Err = results[1] #- std[values]/10 # considering equations plus the standard deviation of parameters divide by ten

        print("ERROR value: " + str(Err)  +" STD/10: "+str(std[values]/10))

        if len(Errstore) == 0:
            greedvar = values
            greedfit = fit
        elif all(Err <= i for i in Errstore): # this is where the accepting happens (CHANGE above error to something more appropriate
            greedvar = values
            greedfit = fit

        Errstore.append(Err)

        layerdic.update({values: [fit, Err]})

    # sets the greedy point as convergence node
    #constants = deepcopy(constantshold)
    var = deepcopy(varibleshold)
    #constants.append([greedvar, concdic.get(
        #greedvar)])  # this will need to be replaced by something in the sql or logic in real example

    # readjusts the scaled parameters
    scaledparas, functparas, funcvar_holder, var,data = listmodifier(greedvar,scaledparas,functparas,funcvar_holder,var,data,mean)

    print(var)
    print(greedvar)
    return greedfit,var,scaledparas,functparas,funcvar_holder,data

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
            data = std_meandatapinned(data, mean[values],var.iloc[values],spaces)

    """somefunction to pin the FUNC PARAMETERS"""

    return stdind,data

def std_meandatapinned(data,mean,varslice,spaces): #logdependance eiter "log" or "notlog"

    #set up parameters to lazy use var to data
    paras = [mean,varslice.iloc[4]]
    print(mean)
    #
    print(varslice)
    x = data.iloc[int(varslice.iloc[5])]
    print(x)
    s = Var2data(x, paras,spaces)

    data.iloc[int(varslice.iloc[5])] = s
    print(s)
    #pu s into data
    return data

# modifies all the stuff used by the optimiser when the parameter is pinned
def listmodifier(value,scaledparas,functparas,funcvar_holder,var,data,mean):
    # gets the cut row
    varsslice = var.iloc[value]
    print("cut varible ")
    print(varsslice)

    value = value + 1 # conversion from starts at zero used by computer to sttarts at 1 used by computer

    if scaledparas[0][0] != 0:
        x = deepcopy(scaledparas)
        ii = 1
        Ns = scaledparas[0][0]
        for stuff in scaledparas[1:]:
            if value == stuff[1]:
                
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
                x[ii][3] = varsslice.iloc[0]
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
                    y = [x[ii][0],x[ii][3],0,varsslice.iloc[0]]
                    x[ii] = y
                else:
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

    return scaledparas,functparas,funcvar_holder, var, data


# modifies all the stuff used by the optimiser when the parameter is pinned
# COPY OF BEFORE I ADDED THE DATA MODIFIfeirs in
def COPYlistmodifier(value,scaledparas,functparas,funcvar_holder,var,data,mean):
    # gets the cut row
    varsslice = var.iloc[value]
    print("cut varible ")
    print(varsslice)

    value = value + 1 # conversion from starts at zero used by computer to sttarts at 1 used by computer

    if scaledparas[0][0] != 0:
        x = deepcopy(scaledparas)
        ii = 1
        Ns = scaledparas[0][0]
        for stuff in scaledparas[1:]:
            if value == stuff[1]:
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
                x[ii][3] = varsslice.iloc[0]
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
                    y = [x[ii][0],x[ii][3],0,varsslice.iloc[0]]
                    x[ii] = y
                else:
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

    return scaledparas,functparas,funcvar_holder, var, data