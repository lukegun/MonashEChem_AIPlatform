import seaborn as sns
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

# loads the bayesian probability of the function
def clusterbayesloader(filename):
    f = open(filename)
    filelist = f.readlines()
    listofdic = []
    for lines in filelist:
        dic = {}
        x = lines.split(" ")
        for i in range(0,6):
            y = x[i*2+1].strip("  \n")
            print(y)
            #if y[0] != "": # check to see if end has been reached
            print(y)
            dic.update({x[i*2]:float(y)})
        listofdic.append(dic)

    return listofdic

ACmode = False

#This is the cluster
modelfile = "DCclustersettings_dtwC5"
value = 5 # number of clusters

fogsize = (16,8)
fontsize = 15
#barycenter = model.cluster_centers_[modellabel].ravel()

dname = ["postgres", "password", "host", port, "table"]

arrs = []
count = 0
arrlabels = []

# reads the data for the bayessetting
#open file
clusterbayesloc = modelfile+ "/BayesoutputClust.txt"

f  = open(modelfile+ "/BayesoutputClust.txt","r")


clusterbayes = clusterbayesloader(clusterbayesloc)

"""LOAD IN THE CLUSTERING FILE"""
if type(clusterbayesloc) != None:
    clusterbayes = clusterbayesloader(clusterbayesloc)
    print(clusterbayes)
    bayesmatrix = np.zeros((len(clusterbayes[0]),len(clusterbayes)))
    for i in range(len(clusterbayes)):
        j = 0
        for key,items in clusterbayes[i].items():
            bayesmatrix[j,i] = items
            j += 1

clo = plt.rcParams['axes.prop_cycle'].by_key()['color']
#print(clo)
clo.append('m')
print(clo)
clo.reverse()
print(clo)
r,c = bayesmatrix.shape

clo = plt.rcParams['axes.prop_cycle'].by_key()['color']
print(clo)
x = []
colour = [clo[0],clo[1],clo[4],clo[3],clo[2]]


data = DataFrame(clusterbayes)
font = 18
print(data.axes[1])
we = data.axes[1]
data = data.transpose()
data["RM"] = ['$E$', '$EC$', '$EE$', '$ECE$', '$E_{Surf}$', '$E_{Cat}$']
print(data)
plt.figure()
data.plot( x="RM", kind="bar",color=colour,figsize=(10,5),fontsize=font)
plt.legend([i for i in range(1,12)],title="$N_c$")
#sns.barplot(data=data,x="RM",color=clo1)
plt.ylim([0,1])
plt.ylabel("Probability $P(M_i|C_j)$",fontsize=font)
plt.xlabel("Reaction Mechanism",fontsize=font)
plt.savefig("Clusterprobability.png",bbox_inches='tight')
