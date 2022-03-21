# A class used to track the priors in the system and acts as a general policy of the optimiser
# Everythiung is hard coded in as a seperate function
import numpy as np


class optpriortracker():

    def __init__(self,reactmech,var):
        self.reactmech = reactmech[0]
        self.initvar = var

        Nparameter = var.shape[0] # number of original parameters

        # saves all para codes
        Nparameter = var.shape[0]  # number of original parameters
        self.dicpara = {}
        for i in range(Nparameter):
            x = [var.iloc[i][4],var.iloc[i][5],var.iloc[i][3],0] # [code,line,logsensitive,number of shifts

            self.dicpara.update({str(var.iloc[i][4]) + "-" + str(var.iloc[i][5]):x})

        # something to modify parameters if theres a chemical step
        if self.reactmech == "EC":
            self.chemicalstep = True
        else:
            self.chemicalstep = False

        self.pararules = {"11":1,"22":2,"33":3,"34":1,"35":2,"41":1,"42":2}

    def priorchange(self,var,rawstd,std,rawarray):

        # identifies if the cma es has optimised out of range
        Nex = 0.97
        extot= []
        for values in rawarray:
            ex = []
            for float in values:
                if float > Nex:
                    ex.append(1)
                elif float < -Nex:
                    ex.append(-1)
                else:
                    ex.append(0)
            extot.append(ex)

        # sets a range of instructions for each parameter passed
        instruct = []
        noshift = []
        i = 0
        for values in extot:
            nval = len(values)
            npos = sum([i== 1 for i in values])
            nneg = sum([i == -1 for i in values])
            if npos == 0 and nneg == 0: # case for when priors are fine
                instruct.append("none")
                noshift.append(i)
            elif npos == nval: # case where priors need to be more positive
                instruct.append("pos")
            elif nneg == nval: # case where priors need to be more negitive
                instruct.append("neg")
            elif npos == nval - 1 and nneg != 1:
                instruct.append("smallpos")
            elif nneg == nval - 1 and npos != 1:
                instruct.append("smallneg")
            else:
                instruct.append("none")
                noshift.append(i)
            i += 1

        # tells the algorithm that priors are shifted
        # its a self object as things might change latter on asw prior shift is rejected
        """self.modprior = []
        print(noshift)
        exit()
        for i in range(var.shape[0]):
            if any([i == j for j in noshift]):
                self.modprior.append(True)
            else:
                self.modprior.append(False)"""
        """NEED SOMETHING TO COLLECT ALL PARAMETERS AND CHANGES"""

        #premetivly checks to see if any varibles have been changed to much and require pinning
        twobpinned = []
        for i in range(var.shape[0]):
            code = var.iloc[i][4]
            row = var.iloc[i][5]
            x = self.dicpara.get(str(code) + "-" + str(row))
            print(code)
            print(self.pararules)
            if x[3] >= self.pararules.get(str(int(code))):
                twobpinned.append(True)
            else:
                twobpinned.append(False)

        #modifies var with changes
        var = self.policy(var,instruct)

        return var, noshift,twobpinned

    # parameter policy for the EE reaction mechanism
    def policy(self,var,instruct):

        for i in range(len(instruct)):
            code = var.iloc[i][4]
            row = var.iloc[i][5]
            x = self.dicpara.get(str(code) + "-" + str(row))

            if instruct[i] != "none" and x[3] < self.pararules.get(str(int(code))):
                code = var.iloc[i][4]
                row = var.iloc[i][5]
                x = self.dicpara.get(str(code)+"-"+str(row))
                print(x)
                if instruct[i] == "pos" or instruct[i] == "smallpos":
                    if x[2] == 1: # log parameter
                        diff = np.log10(var.iloc[i][2]) - np.log10(var.iloc[i][0])

                    else:   # not log parameter
                        diff = var.iloc[i][2] - var.iloc[i][0]

                elif instruct[i] == "neg" or instruct[i] == "smallneg":
                    if x[2] == 1:  # log parameter
                        diff = np.log10(var.iloc[i][1]) - np.log10(var.iloc[i][0])
                    else:  # not log parameter
                        diff = var.iloc[i][1] - var.iloc[i][0]


                # sets up the adjustment
                if x[2] == 0:
                    newprior = [var.iloc[i][0] + diff,var.iloc[i][1] + diff,var.iloc[i][2] + diff]
                else:
                    a = 10 ** (np.log10(var.iloc[i][0]) + diff)
                    b = 10 ** (np.log10(var.iloc[i][1]) + diff)
                    c = 10 ** (np.log10(var.iloc[i][2]) + diff)
                    newprior = [a,b,c]
                print(newprior)
                if int(code/10) == 1: #Resistance
                    if code == 11:
                        newprior = self.bounder(newprior, 1, x)
                elif int(code/10) == 2: # diffussion and concentration
                    if code == 21: #concentration
                        print("ERROR: concentration shifting not been set up ")
                        exit()
                    else: #diffusion
                        newprior = self.bounder(newprior, newprior[1]/10, x)

                elif int(code/10) == 3: # reaction parameters
                    if code == 31 or code == 32: # kb and kf
                        print("ERROR: kb/kf shifting not been set up ")
                        exit()
                    elif code == 33:
                        pass
                        #all values are possible

                    elif code == 34:
                        newprior = self.bounder(newprior,0.0001,x)
                        # adjustment for EC systems
                        if not self.chemicalstep:
                            newprior = self.boundergreater(newprior, 10, x)

                    elif code == 35:
                        newprior = self.bounder(newprior, 0.3, x)
                        newprior = self.boundergreater(newprior, 0.7, x)

                elif code == 41: # functional reaction parameters
                    newprior = self.bounder(newprior, 0.001, x)

                elif code == 42:
                    newprior = self.bounder(newprior, 0.001, x)
                    newprior = self.boundergreater(newprior, 89.999, x)

                # below works but this
                # also needs to be
                var.loc[var.index[i],0] = newprior[0]
                var.loc[var.index[i],1] = newprior[1]
                var.loc[var.index[i],2] = newprior[2]

                #update the prior shift
                x[3] = x[3] + 1 # saves that the modifiers have been modiefied once
                self.dicpara.update({str(code) + "-" + str(row):x}) #updates the saved dictionary

            else:
                pass

        return var

    #checks to see if value is smaller then limit then adjusts
    def bounder(self,newprior,value,x):

        if newprior[1] < value:
            newprior[1] = value#0.0001  # just sets the minimum value of a tenth of the mean
            if x[2] == 0:
                newprior[0] = (newprior[1] + newprior[2]) / 2
            else:
                newprior[0] = 10 ** ((np.log10(newprior[1]) + np.log10(newprior[2])) / 2)

        return newprior

    # checks to see if value is greater then limit then adjusts
    def boundergreater(self, newprior, value, x):

            if newprior[2] > value:
                newprior[2] = value  # 0.0001  # just sets the minimum value of a tenth of the mean
                if x[2] == 0:
                    newprior[0] = (newprior[1] + newprior[2]) / 2
                else:
                    newprior[0] = 10 ** ((np.log10(newprior[1]) + np.log10(newprior[2])) / 2)

            return newprior
