import scipy as sci
from scipy.special import jv, yn    # jv = first order, yn =second kind
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time


Nc = [4,5,6,7,8]
sak = False

"""MODEL LOCATION"""

modellocsak = ["DCclustersettings_dtwC4","DCclustersettings_dtwC5","DCclustersettings_dtw_saiko64",
                "DCclustersettings_dtwC7","DCclustersettings_dtwC8"]


s = "dtwcluster_nosaiko/"
modellocdtw = [s+"DCclustersettings_dtwC4",s+"DCclustersettings_dtwC5","DCclustersettings_dtw_V1"
        ,s+"DCclustersettings_dtwC7",s+"DCclustersettings_dtwC8"]

modellocfull = [modellocdtw,modellocsak]

plt.figure()
modelarrlabels = []
for modelloc in modellocfull:
    arrs = []
    count = 0
    arrlabels = []

    for value in Nc:
        #open file
        print("attempt")
        print(modelloc[count])
        print(value)
        f  = open(modelloc[count]+ "/Bayesoutput.txt","r")

        # load the data up
        lines = f.readlines()
        # This is simpilist way to count the number of reaction mechanisms sadly
        Nr = 0
        for j in range(int(len(lines)/2)):
            if lines[3 + j*2][0:5] == "label":
                Nr += 1
            else:
                break

        labelfreq = np.zeros((value,Nr))
        labelprob = np.zeros((value,Nr))
        classnumber = np.zeros((value))

        # collect label frequency
        nn = 4
        for j in range(Nr):
            for jj in range(value):
                labelfreq[jj,j] = float(lines[nn+2*j].split(" ")[2+ 3*(jj)])

        nn = nn+2*Nr + 3
        for j in range(value):
            for jj in range(Nr):
                labelprob[j,jj] = float(lines[nn+2*j].split(" ")[1+ 2*jj])


        nn  = nn +2*value + 2
        print(lines[nn])
        for j in range(value):
            print(lines[nn+j])
            classnumber[j] = int(lines[nn+j].split(" ")[1])

        #print(modelloc[i])
        #print(labelprob)
        #print(labelfreq)
        #print(classnumber)
        #print(abs(labelfreq[i, :] - labelfreq[j, :]))
        sumer = 0
        for i in range(value):
            sumerint = 0
            for j in range(value):
                #print(j)
                f = 0
                if j>i:
                    print(len(labelfreq[i,:]),value)
                    sumerint += sum(abs((labelfreq[i,:]-labelfreq[j,:]))**2/len(labelfreq[i,:]))
                    f += 1
                #print(i)
                #print(abs((labelfreq[i,:]-labelfreq[j,:]))**2/Nr)
            #print((value-i))

            if f != 0:
                print(f)
                sumer += sumerint/(value)
        arrlabels.append(sumer)

            # put into matric
        #for j in range(Nc[i]):


        count += 1
    modelarrlabels.append(arrlabels)

print(modellocfull)
font = 18
labels = ["dtw","saiko64"]
plt.figure(figsize=(8,5))
for i in range(len(modelarrlabels)):
    print(modelarrlabels[i])
    plt.scatter(Nc,modelarrlabels[i],label =labels[i],linewidth=5,linestyle='None',marker="h") #linewidth=5,

plt.xlabel("$N_c$",fontsize=font)
st  = r'$\Psi$'
plt.ylabel(st,fontsize=font)
plt.xlabel("$N_c$",fontsize=font)
plt.locator_params(axis='x', nbins=5)
plt.tick_params(axis='both', labelsize=16)
plt.legend(fontsize=font)
plt.savefig("Ncaccuracy_value.png",bbox_inches='tight')
