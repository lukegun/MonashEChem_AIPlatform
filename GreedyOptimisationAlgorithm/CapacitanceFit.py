
import numpy as np
import mecsim
from scipy import optimize as scipyopt
from scipy.signal import argrelextrema
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from copy import deepcopy
import time


class overallcapfit():

    def __init__(self,Excurr,ACmode,trun,dt,Evalues,scanrate,expvoltage,fundimentalharm,ACsettings):

        """SOMETHING TO DEFINE WHICH  MODEL"""

        self.Excurr = Excurr

        self.ACmode = ACmode # True means AC
        if ACmode:
            self.fundimentalharm = fundimentalharm
            self.ACsettings = ACsettings

        # experimental truncation
        self.truncation = trun

        # required for gradient functions
        self.dt = dt

        #Evalues = [Estart, Eend, Emax, Emin, Ncycles]
        # voltage specification here
        self.DCvoltage = DCsetter(Evalues ,scanrate,len(Excurr),expvoltage)
        self.expvoltage = expvoltage
        self.scanrate = scanrate
        """ DC OR AC CAP FIT """
        self.E0value = [] # holds the Eapparent value

        self.deci = int(len(self.Excurr)/16384)


    def overall_fitting(self):

        #"""EXTRACT THE SPECIFIC DATA"""
        capregion = self.automatic_regionselect()

        """extract region defined as the capacitance region from voltage, time and current for fitting DO DECIMATION HERE"""
        deci = self.deci # should be calculated required to make things efficent
        DCvoltage = self.DCvoltage[::deci]
        expvoltage = self.expvoltage[::deci]
        fundimentalharm = self.fundimentalharm[::deci]

        #DChold = []
        evolthold = []

        # this derivative has no effect on length
        dedt = np.gradient(expvoltage, edge_order=2) / (self.dt*deci)  # 1st derivative
        didt = np.gradient(fundimentalharm, edge_order=2) / (self.dt*deci)

        currenthold = []
        dedthold = []
        didthold = []
        for values in capregion:

            #DChold.append(DCvoltage[int(values[0]/deci):int(values[1]/deci)])
            evolthold.append(expvoltage[int(values[0]/deci):int(values[1]/deci)])
            dedthold.append(dedt[int(values[0]/deci):int(values[1]/deci)])
            didthold.append(didt[int(values[0]/deci):int(values[1]/deci)])
            currenthold.append(fundimentalharm[int(values[0]/deci):int(values[1]/deci)])

        """put in something so that if regions are less of equal to two "unbounded cdl = c0"""
        if len(capregion) == 2 or len(capregion) == 1:
            Ncdl = 1
        elif len(capregion) == 0: # in this case set everything to zero and ignore
            print("Error no cap region identified canceling process")
            Ncdl = 0
        else:
            Ncdl = 5
        
        """MIGHT PUT IN SOME POLYNOMIAL REDUCTION for Cdl"""
        #C coefficents need to be purely in farads
        bounds = []
        """CAPICTANCE OPTIMSATION """
        if Ncdl != 0:
            pass
            tolx = None
            x0 = [10*max(currenthold[0])/(self.scanrate+self.ACsettings[0]*2*np.pi*self.ACsettings[1]/1000)]
            bounds.append((0,100*x0[0]))

            for i in range(Ncdl - 1): # add starting points for others
                x0.append((-1)**(i+1)*x0[0])
                bounds.append((-100*x0[0], 100*x0[0]))

            x0.append(500) # add resistance
            bounds.append((0.1, 5000))

            self.voltage = evolthold
            self.current = currenthold
            self.dedt = dedthold
            self.didt = didthold
            self.Ncap = Ncdl

            # do the optimisation
            output = minimize(self.capmodel, x0, args=(), method='Nelder-Mead', bounds=bounds, tol=tolx)
            print(output)

            # gets the IDcl
            paras = output.get("x")

            V = expvoltage[:] - fundimentalharm[:] * paras[-1]

            Cdl = np.ones(len(expvoltage)) * paras[0]
            if self.Ncap != 1:
                for i in range(1, self.Ncap):
                    Cdl += paras[i] * V[:] ** (i)

                # calculates the double layer derivative
                DCdlDV = np.ones(len(expvoltage)) * paras[1]
                for i in range(2, self.Ncap):
                    DCdlDV += (i) * paras[i] * V[:] ** (i - 1)
            else:  # case if Cdl is constant
                DCdlDV = np.zeros(len(expvoltage))

            Icdlfit = (Cdl[:] + V[:] * DCdlDV[:]) * (dedt[:] - paras[-1] * didt[:])

        else:
            #sets all simulated values as Zero
            capparas = [0,0,0,0,0,0]
        # calculate an initial value

        plt.figure()
        # for j in range(len(voltage)):
        plt.plot(fundimentalharm[100:])
        plt.plot(Icdlfit[100:])
        plt.savefig("captest.png")
        plt.close()

        if len(self.E0value) != 0 or len(self.E0value) != 1:  # move this to out side function
            E0values = self.checkevalue()
        else:
            print("no faradaic component detetected in automated fitting")
            E0values = None

        # use some polynomial regresssion


        """plot the model background and the experimental data and the MECSIM output assuming some R value from above"""


        return paras,Icdlfit, E0values

    # checks for duplicates applied potential in finding
    def checkevalue(self):

        l = []

        Evalues = deepcopy(self.E0value)
        i = 0
        while i != len(Evalues):
            l.append(self.E0value[i])
            j = len(l)
            while j < len(Evalues):
                if abs(l[i] - Evalues[j]) < 0.025: # if the difference is less then 25 mvassume its the same
                    l[i] = (l[i] + Evalues[j])/2 # assume the average is a good approx of E apparent
                    Evalues.pop(j)
                j += 1
            i +=1

        return l


    def capmodel(self, *args):

        paras = args[0]

        # paras of form
        # paras = [c0,c1,c2,c3,c4,Ru] or [c0,c1,c2,Ru] or so on

        """This probally all needs to be done in one giant loop for each region"""
        j = 0
        Cdlhold = []
        DCdlDVhold = []
        Vhold = []
        for j in range(len(self.voltage)):
            """NEED TO TRUNCATE THE ENDS TO MATCH UP WITH THE DERIVATIVE"""
            V = self.voltage[j][:] - self.current[j][:]*paras[-1]
            Vhold.append(V)

            Cdl = np.ones(len(self.voltage[j]))*paras[0]
            if self.Ncap != 1:
                for i in range(1,self.Ncap):
                    Cdl += paras[i]*V[:]**(i)

                # calculates the double layer derivative
                DCdlDV = np.ones(len(self.voltage[j])) * paras[1]
                for i in range(2,self.Ncap):
                    DCdlDV += (i)*paras[i] * V[:] ** (i-1)
            else: # case if Cdl is constant
                DCdlDV = np.zeros(len(self.voltage[j]))

            #Add to list of capacitance stuff
            Cdlhold.append(Cdl)
            DCdlDVhold.append(DCdlDV)

        # calculates the capacitance current
        Icdlhold = []
        for j in range(len(self.voltage)):
            Icdl = (Cdlhold[j][:] + Vhold[j][:]*DCdlDVhold[j][:])*(self.dedt[j][:]- paras[-1]*self.didt[j][:])
            Icdlhold.append(Icdl)
        
        x = 0
        for j in range(len(self.voltage)):
            x += sum((Icdlhold[j][:] - self.current[j][:])**2)/len(Icdlhold[j][:])

        return x


    def automatic_regionselect(self):

        if self.ACmode:
            print("need to include the truncation and decimation on the DC voltagage")
            x = self.Excurr[self.truncation[0]:int(len(self.Excurr)-self.truncation[0])]

            n = len(x)
            gradhold = [x]

            # get the recursive gradients
            """cumsum takes the (window_width-1) from the array so these points will needed to be added back {ASSUME FLAT}"""
            for i in range(2):
                #plt.figure()
                x = np.gradient(x, edge_order=2) / self.dt  # 1st derivative
                #plt.plot(x)
                # cheap line smoothing incase gradient is noisy using cumsum
                window_width = int(len(x)/100)

                cumsum_vec = np.cumsum(np.insert(x, 0, 0))
                x = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
                # bellow was added for edge cases

                x = np.concatenate((np.ones(int(round(window_width, 0)/2))*x[0], x, np.ones(int(window_width/2))*x[-1]))

                #plt.plot(x)
                # done twice to make sure system is good done the opisite way to undo shift in values
                #np.convolve(x, np.ones(n) / n, mode='valid')
                cumsum_vec = np.cumsum(np.insert(x, 0, 0))
                x = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
                x = np.concatenate((np.ones(int(round(window_width, 0)/2)-1)*x[0], x, np.ones(int(window_width/2 ))*x[-1]))

                gradhold.append(x)

            # we can probably truncate this stuff


            #NEEDS SOMETHINGS SO THAT True points is included
            n = 2**5

            xhold = []
            for i in range(0,3):
                x = gradhold[i]
                x = x[::n]
                window = int(len(x) / (50)) # this needs to be a parameter
                steadypointg = argrelextrema(x, np.greater, axis=0, order=window, mode='clip')
                steadypointl = argrelextrema(x, np.less, axis=0, order=window, mode='clip')

                xg = []
                for values in steadypointg[0]:
                    xg.append(values * n)  # +self.truncation[0])

                xl = []
                for values in steadypointl[0]:
                    xl.append(values * n)  # +self.truncation[0])
                xhold.append([xl,xg])

            plt.figure()
            fig, axs = plt.subplots(3,figsize=(15,15))
            for i in range(3):
                axs[i].plot(gradhold[i])#[self.truncation[0]:int(int(len(gradhold[i])/2)-self.truncation[0])]

            for i in range(3):

                #func
                axs[i].vlines(xhold[0][1], min(gradhold[i]), max(gradhold[i]), colors='k')

                #1st
                axs[i].vlines(xhold[1][0], min(gradhold[i]), max(gradhold[i]), colors='m')
                axs[i].vlines(xhold[1][1], min(gradhold[i]), max(gradhold[i]), colors='y')

                #2nd
                axs[i].vlines(xhold[2][0], min(gradhold[i]), max(gradhold[i]), colors='b')
                axs[i].vlines(xhold[2][1], min(gradhold[i]), max(gradhold[i]), colors='r')

            plt.savefig("gradtest.png")
            plt.close()
            # the below is horrible but it works
            dichold = {}
            jj = 0
            for values in xhold[0][1]:
                dichold.update({"fm"+str(jj):values})
                jj += 1

            array = ["1l","1m","2l","2m"]
            j = 1
            i = 0
            while j != 3:
                for stuff in xhold[j]:
                    jj = 0
                    for values in stuff:
                        dichold.update({array[i]+str(jj): values})
                        jj += 1
                    i += 1
                j += 1

            #code to identify the faradaic region
            dictoy = {k: v for k, v in sorted(dichold.items(), key=lambda item: item[1])}

            # CHECK THE FUNDIMENTAL MAX IN THE CORRECT PATTERN
            i = 0
            startc = False
            v = ""
            arraylist = []
            fregionlist = []
            for values, items in  dictoy.items():
                if startc or i == 5:
                    startc = True
                    v = v + values[:2]
                    arraylist.append(items)
                    if v == "2m1m2lfm1l2m" or v == "2m1mfm2l1l2m" or v == "2m1l2lfm1m2m" or v == "2m1l2lfm1m2m": # last two are the flipped conditionsconditions
                        print("faradaic component found")

                        #below are done for blurring bassed on assumitry in the system
                        scale = 1.5 # scale is added to proper avoid the faradiac region
                        diffleft = int(scale*(arraylist[1] - arraylist[0]))
                        diffright = int(scale*(arraylist[-1] - arraylist[-2]))

                        # allocates the mid point
                        if v[6:8] == "fm":
                            midpoint = arraylist[3] + self.truncation[0]

                        elif v[4:6] == "fm":
                            midpoint = arraylist[2] + self.truncation[0]

                        else:
                            # error will continue and hopefully no issue
                            print("Error in function allocation mid point")
                            midpoint = arraylist[3] + self.truncation[0]

                        self.E0value.append(self.DCvoltage[midpoint])

                        """print(arraylist[-1] - arraylist[0])
                        print(len(self.Excurr)/5)
                        # check to see if faradaic component is an error or capacitance
                        if arraylist[-1] - arraylist[0] > len(self.Excurr)/5: # theres better ways but this is well rounded for now"""
                        # above didn't work tryusing the current on fm Format [left, right, mid]

                        fregionlist.append([self.truncation[0] + arraylist[0]-diffleft, self.truncation[0] + arraylist[-1]+diffright,midpoint])

                    # this is done for preperation of next loop
                    v = v[2:]
                    arraylist.pop(0)


                else:
                    v += values[:2]
                    arraylist.append(items)

                i += 1

            # will need something here incase above fails

            #checks to see if the switching potential has been labelled by the faradaic component
            Nmid = len(self.Excurr)/2
            nmidlow = Nmid - 2*Nmid/100
            nmidhigh = Nmid + 2*Nmid/100

            faradic = deepcopy(fregionlist)
            i = 0
            for values in faradic:
                if values[2] > nmidlow and  values[2] < nmidhigh:
                    fregionlist.pop(i) # deletes the entry that is the sitching potential
                i += 1


        else:
            """Base this one off the half peak potentials and the """
            pass


        # Add the capicity region with some loose line definition regions
        capregion = []
        n = len(fregionlist)
        for i in range(n + 1):
            if i == 0: # empty
                capregion.append([self.truncation[0],fregionlist[0][0]])
            elif i != n:
                capregion.append([fregionlist[i-1][1], fregionlist[i][0]])
            else:
                capregion.append([fregionlist[i-1][1], len(self.Excurr)-self.truncation[0]])

            # some general checks
            if capregion[-1][0] >= capregion[-1][1]: #deletes if the region is so small the smudging made it flip
                capregion.pop(-1)

            elif int(capregion[-1][1] - capregion[-1][0]) < int(len(self.Excurr)/100): # if region is smaller then 1% delete it
                capregion.pop(-1)

        #x = np.linspace(0, len(self.Excurr), len(self.Excurr))
        plt.figure()
        plt.plot(self.Excurr)
        for values in capregion:
            plt.axvspan(values[0], values[1], alpha=0.5, color='red')
        plt.savefig("testregion.png")
        plt.close()


        # combine to the background regions with the voltage and pass to optimisation


        return capregion

    def cap_plot(self):

        return

# function for finding the nearest value in a series of array
def find_nearest(array, value):
    ''' Find nearest value is an array '''
    idx = (np.abs(array-value)).argmin()
    return idx

def DCsetter(Evalues,scanrate,Np,Expvoltage):

    minidx = find_nearest(Expvoltage,min(Expvoltage))
    maxidx = find_nearest(Expvoltage,max(Expvoltage))

    if Evalues[0] == Evalues[2] and Evalues[0] == Evalues[1]: # starts at max value and goes negitive
        case = 0
    elif Evalues[0] ==Evalues[3] and Evalues[0] == Evalues[1]:  # starts at min value and goes positive
        case = 1
    elif minidx < maxidx: # starts goes negitive then goes positive then goes to end
        case = 2
    elif maxidx < minidx:  # starts goes pos then goes neg then goes to end
        case = 3

    Ncycle = int(Np/Evalues[4])
    # Evalues = [Estart, Eend, Emax, Emin, Ncycles]

    if case == 0:
         minsweep = np.linspace(Evalues[0],Evalues[3],num=int(Ncycle/2))
         maxsweep = np.linspace(Evalues[3],Evalues[0],num=int(Ncycle/2))
         Vcycle = np.concatenate((minsweep,maxsweep))
    elif case == 1:
        minsweep = np.linspace(Evalues[0], Evalues[2], num=int(Ncycle / 2))
        maxsweep = np.linspace(Evalues[2], Evalues[0], num=int(Ncycle / 2))
        Vcycle = np.concatenate((minsweep, maxsweep))
    elif case == 2:

        # below is rough but should work enough
        n1 = abs(Evalues[3]-Evalues[0])/scanrate
        n2 = abs(Evalues[2]-Evalues[3])/scanrate
        n3 = abs(Evalues[1]-Evalues[2])/scanrate
        ntot = n1+n2+n3

        Np1 = int(Np*(n1/ntot))
        Np2 = int(Np*(n2/ntot))
        Np3 = int(Np * (n3 / ntot))

        minsweep1 = np.linspace(Evalues[0], Evalues[3], num=Np1)
        maxsweep1 = np.linspace(Evalues[3], Evalues[2], num=Np2)

        minsweep2 = np.linspace(Evalues[2], Evalues[1], num=Np3)

        Vcycle = np.concatenate((minsweep1, maxsweep1,minsweep2))

    elif case == 3:

        n1 = abs(Evalues[2] - Evalues[0]) / scanrate
        n2 = abs(Evalues[2] - Evalues[3]) / scanrate
        n3 = abs(Evalues[1] - Evalues[3]) / scanrate
        ntot = n1 + n2 + n3

        Np1 = int(Np * (n1 / ntot))
        Np2 = int(Np * (n2 / ntot))
        Np3 = int(Np * (n3 / ntot))

        minsweep1 = np.linspace(Evalues[0], Evalues[2], num=Np1)
        maxsweep1 = np.linspace(Evalues[2], Evalues[3], num=Np2)

        minsweep2 = np.linspace(Evalues[3], Evalues[1], num=Np3)

        Vcycle = np.concatenate((minsweep1, maxsweep1, minsweep2))

    fullcycle = Vcycle
    if int(Ncycle) != 1:
        for i in range(int(Evalues[4]-1)):
            fullcycle = np.concatenate(fullcycle, Vcycle)

    s = 1/1000 # millivolt to volt
    DCvoltage = fullcycle*s

    return DCvoltage

