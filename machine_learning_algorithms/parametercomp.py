import scipy as sci
from scipy.special import jv, yn    # jv = first order, yn =second kind
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
import string
import psycopg2
from tslearn.clustering import TimeSeriesKMeans

#This is the cluster
modelfile = "DCclustersettings_dtwC5"
value = 5 # number of clusters

fogsize = (16,8)
fontsize = 26
linefont = 5

model = TimeSeriesKMeans.from_hdf5(path=modelfile+"/trainedtimeseries-1.hdf5")
#barycenter = model.cluster_centers_[modellabel].ravel()

dname = ["postgres", "password", "host", port, "table"]

arrs = []
count = 0
arrlabels = []

# reads the data for the bayessetting
#open file
f  = open(modelfile+ "/Bayesoutput.txt","r")

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
    classnumber[j] = int(lines[nn+j].split(" ")[1])

f.close()

# load up the classifier label
f = open(modelfile+ "/ReactionIDlabels.txt","r")
# load the data up
lines = f.readlines()
stringsstuff = string.ascii_letters
newlines = []
for l in lines:
    x = l.split("\t")
    s = x[1].strip("\n")
    for i in range(12):
        # converts strings labels to int
        if stringsstuff[i] == s[-1]:
            newlines.append([int(x[0]),i])

try:
    connection = psycopg2.connect(user=dname[0],
                                      password=dname[1],
                                      host=dname[2],
                                      port=dname[3],
                                      database=dname[4])

    cursor = connection.cursor()
    para = cursor.execute("""SELECT "Reaction_ID", "formalE","Ru" FROM "Simulatedparameters" WHERE "ReactionMech" = %s""", ("EE",))
    para = []
    for table in cursor.fetchall():
        para.append(table)
    print(para)
    # sqlcom = """INSERT INTO time(test,"Ctime") VALUES( %()s, CURRENT_TIMESTAMP )"""

except (Exception, psycopg2.Error) as error:
    print("error,", error)
finally:
        # This is needed for the
    if (connection):
        cursor.close()
        connection.close()

print(newlines[0],para[0])
newnewline = []
for l in newlines:
    for k in para:
        if l[0] == k[0]:
            newnewline.append([l[0],l[1],k[1],k[2]])
            break


# modification to calc deltaE
for i in range(len(newnewline)):
    newnewline[i][2] = newnewline[i][2][1] - newnewline[i][2][0]
    newnewline[i][3] = newnewline[i][3]#[1]# + newnewline[i][3][0])*0.5

#print(newnewline)
ptot = []
for i in range(value):
    p = []
    for l in newnewline:
        if i == l[1]:
            p.append([l[2],l[3]])
    ptot.append(p)

x = []
fig, axs = plt.subplots(1, 2, tight_layout=True,figsize=fogsize)
i = 0
for l in ptot:
    print(len(l))
    if len(l) > 100:
        ll = np.array(l)
        print(ll.shape)
        #axs[0].scatter(ll[:, 0], ll[:, 1], alpha=0.5)
        #axs[0].set_yscale("log")
        axs[0].hist(ll[:, 0],50,alpha=0.5)
        axs[1].plot(model.cluster_centers_[i].ravel(),linewidth=linefont)
        x.append(len(l))
    i += 1
axs[1].legend(x,loc='lower right',title="Number of\nsimulations",fontsize=fontsize,title_fontsize=fontsize)
axs[0].set_xlabel("$\Delta E$",fontsize=fontsize)
axs[0].set_ylabel("Frequency",fontsize=fontsize)
axs[1].set_xlabel("Datapoint",fontsize=fontsize)
axs[1].set_ylabel("Timeseries Averaged Currents",fontsize=fontsize)

fig.suptitle("$EE$",fontsize=fontsize*2)
fig.savefig("deltaE.png")

try:
    connection = psycopg2.connect(user=dname[0],
                                      password=dname[1],
                                      host=dname[2],
                                      port=dname[3],
                                      database=dname[4])

    cursor = connection.cursor()
    para = cursor.execute("""SELECT "Reaction_ID", "kbackward", "kforward" FROM "Simulatedparameters" WHERE "ReactionMech" = %s""", ("EC",))
    para = []
    for table in cursor.fetchall():
        para.append(table)
    print(para)
    # sqlcom = """INSERT INTO time(test,"Ctime") VALUES( %()s, CURRENT_TIMESTAMP )"""

except (Exception, psycopg2.Error) as error:
    print("error,", error)
finally:
        # This is needed for the
    if (connection):
        cursor.close()
        connection.close()

print(newlines[0],para[0])
newnewline = []
for l in newlines:
    for k in para:
        if l[0] == k[0]:
            newnewline.append([l[0],l[1],[k[1][1],k[2][1]]])
            break


"""# modification to calc deltaE
for i in range(len(newnewline)):
    newnewline[i][2] = newnewline[i][2][1] - newnewline[i][2][0]"""

#print(newnewline)
ptot = []
for i in range(value):
    p = []
    for l in newnewline:
        if i == l[1]:
            x = l[2]
            x.append(l[0])
            p.append(l[2])
    ptot.append(p)


print(ptot)
fig, axs = plt.subplots(1, 3, tight_layout=True,figsize=fogsize)
x = []
plotnum = []
i = 0
for l in ptot:
    print(len(l))
    if len(l) > 100:
        l = np.array(l)
        print(l.shape)
        axs[0].scatter(l[::4,0],l[::4,1], alpha=0.5)
        axs[0].set_xscale("log")
        axs[0].set_yscale("log")
        plotnum.append(l[np.random.randint(l.shape[0]), 2])  # gets a random thing to plot
        axs[1].hist(np.log10(l[:,1]), 50,alpha=0.5)
        #axs[1].set_xscale("log")
        axs[2].plot(model.cluster_centers_[i].ravel(),linewidth=linefont)
        x.append(len(l[:,0]))
    i += 1

axs[2].legend(x,loc='lower right',title="Number of\nsimulations",fontsize=fontsize,title_fontsize=fontsize)
axs[0].set_xlabel("$k_b$",fontsize=fontsize)
axs[0].set_ylabel("$k_f$",fontsize=fontsize)
axs[1].set_xlabel("$Log_{10}(k_f)$",fontsize=fontsize)
axs[1].set_ylabel("Frequency",fontsize=fontsize)
axs[2].set_xlabel("Datapoint",fontsize=fontsize)
axs[2].set_ylabel("Timeseries Averaged Currents",fontsize=fontsize)
fig.suptitle("$EC$",fontsize=fontsize*2)
fig.savefig("EC_paras.png")
plt.close()

try:
    connection = psycopg2.connect(user=dname[0],
                                      password=dname[1],
                                      host=dname[2],
                                      port=dname[3],
                                      database=dname[4])
    i =1
    for num in plotnum:
        print(num)
        cursor = connection.cursor()
        para = cursor.execute("""SELECT "TotalCurrent" FROM "HarmTab" WHERE "Reaction_ID" = %s""", (int(num),))
        for table in cursor.fetchall():
            curr = table[0]
        print(len(curr))
        fsweep = np.linspace(0, 100, num=int(len(curr) / 2))
        bsweep = np.linspace(100, 0, num=int(len(curr) / 2))
        Rx = np.concatenate((fsweep,bsweep))
        plt.figure()
        plt.plot(Rx,curr)
        plt.savefig("samplereact/EC_sample"+str(int(num))+"C"+str(i)+".png")
        i += 1

    print(para)
    # sqlcom = """INSERT INTO time(test,"Ctime") VALUES( %()s, CURRENT_TIMESTAMP )"""

except (Exception, psycopg2.Error) as error:
    print("error,", error)
finally:
        # This is needed for the
    if (connection):
        cursor.close()
        connection.close()



#########################PLOTS THE ECat systems #############################################

try:
    connection = psycopg2.connect(user=dname[0],
                                      password=dname[1],
                                      host=dname[2],
                                      port=dname[3],
                                      database=dname[4])

    cursor = connection.cursor()
    para = cursor.execute("""SELECT "Reaction_ID", "Conc", "kforward" FROM "Simulatedparameters" WHERE "ReactionMech" = %s""", ("ECat",))
    para = []
    for table in cursor.fetchall():
        para.append(table)
    print(para)
    # sqlcom = """INSERT INTO time(test,"Ctime") VALUES( %()s, CURRENT_TIMESTAMP )"""

except (Exception, psycopg2.Error) as error:
    print("error,", error)
finally:
        # This is needed for the
    if (connection):
        cursor.close()
        connection.close()

print(newlines[0],para[0])
newnewline = []
for l in newlines:
    for k in para:
        if l[0] == k[0]:
            newnewline.append([l[0],l[1],[k[1][-1],k[2][1]]])
            break

"""# modification to calc deltaE
for i in range(len(newnewline)):
    newnewline[i][2] = newnewline[i][2][1] - newnewline[i][2][0]"""

#print(newnewline)
ptot = []
for i in range(value):
    p = []
    for l in newnewline:
        if i == l[1]:
            x = l[2]
            x.append(l[0])
            print(x)
            p.append(x)
    ptot.append(p)


fig, axs = plt.subplots(1, 3, tight_layout=True,figsize=fogsize)
plotnum = []
i = 0
print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
clo = plt.rcParams['axes.prop_cycle'].by_key()['color']
print(clo)
x = []
colour = [clo[0],clo[1],clo[3],clo[2]]
ii = 0
for l in ptot:
    print(len(l))
    if len(l) > 100:
        l = np.array(l)
        print(l.shape)
        axs[0].scatter(l[::2,0],l[::2,1], alpha=0.5,color=colour[ii])
        axs[0].set_xscale("log")
        axs[0].set_yscale("log")
        axs[1].hist(np.log10(l[:,1]), 50,alpha=0.5,color=colour[ii])
        plotnum.append(l[np.random.randint(l.shape[0]),2]) # gets a random thing to plot
        #axs[1].set_xscale("log")
        axs[2].plot(model.cluster_centers_[i].ravel(),color=colour[ii],linewidth=linefont)
        x.append(len(l[:,0]))
        ii += 1
    i += 1

axs[2].legend(x,loc='lower right',title="Number of\nsimulations",fontsize=fontsize,title_fontsize=fontsize)
axs[0].set_xlabel("$C_{Cat}$",fontsize=fontsize)
axs[0].set_ylabel("$k_f$",fontsize=fontsize)
axs[1].set_xlabel("$Log_{10}(k_f)$",fontsize=fontsize)
axs[1].set_ylabel("Frequency",fontsize=fontsize)
axs[2].set_xlabel("Datapoint",fontsize=fontsize)
axs[2].set_ylabel("Timeseries Averaged Currents",fontsize=fontsize)

fig.suptitle("$E_{Cat}$",fontsize=fontsize*2)
fig.savefig("ECatparas.png")
plt.close()

try:
    connection = psycopg2.connect(user=dname[0],
                                      password=dname[1],
                                      host=dname[2],
                                      port=dname[3],
                                      database=dname[4])
    i =1
    for num in plotnum:
        print(num)
        cursor = connection.cursor()
        para = cursor.execute("""SELECT "TotalCurrent" FROM "HarmTab" WHERE "Reaction_ID" = %s""", (int(num),))
        for table in cursor.fetchall():
            curr = table[0]
        print(len(curr))
        fsweep = np.linspace(0, 100, num=int(len(curr) / 2))
        bsweep = np.linspace(100, 0, num=int(len(curr) / 2))
        Rx = np.concatenate((fsweep,bsweep))
        plt.figure()
        plt.plot(Rx,curr)
        plt.savefig("samplereact/ECat_sample"+str(int(num))+"C"+str(i)+".png")
        i += 1

    print(para)
    # sqlcom = """INSERT INTO time(test,"Ctime") VALUES( %()s, CURRENT_TIMESTAMP )"""

except (Exception, psycopg2.Error) as error:
    print("error,", error)
finally:
        # This is needed for the
    if (connection):
        cursor.close()
        connection.close()


#########################PLOTS THE Esurf systems #############################################

try:
    connection = psycopg2.connect(user=dname[0],
                                      password=dname[1],
                                      host=dname[2],
                                      port=dname[3],
                                      database=dname[4])

    cursor = connection.cursor()
    para = cursor.execute("""SELECT "Reaction_ID", "ksreal", "alpha" FROM "Simulatedparameters" WHERE "ReactionMech" = %s""", ("ESurf",))
    para = []
    for table in cursor.fetchall():
        para.append(table)
    print(para)
    # sqlcom = """INSERT INTO time(test,"Ctime") VALUES( %()s, CURRENT_TIMESTAMP )"""

except (Exception, psycopg2.Error) as error:
    print("error,", error)
finally:
        # This is needed for the
    if (connection):
        cursor.close()
        connection.close()

print(newlines[0],para[0])
newnewline = []
for l in newlines:
    for k in para:
        if l[0] == k[0]:
            newnewline.append([l[0],l[1],[k[1][0],k[2][0]]])
            break

"""# modification to calc deltaE
for i in range(len(newnewline)):
    newnewline[i][2] = newnewline[i][2][1] - newnewline[i][2][0]"""

#print(newnewline)
ptot = []
for i in range(value):
    p = []
    for l in newnewline:
        if i == l[1]:
            x = l[2]
            x.append(l[0])
            #print(x)
            p.append(x)
    ptot.append(p)

fig, axs = plt.subplots(1, 2, tight_layout=True,figsize=fogsize)
plotnum = []
i = 0
print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
clo = plt.rcParams['axes.prop_cycle'].by_key()['color']
print(clo)
x = []
colour = [clo[4],clo[1],clo[4],clo[2]]
ii = 0
for l in ptot:
    print(len(l))
    if len(l) > 100:
        l = np.array(l)
        print(l.shape)
        axs[0].scatter(l[::2,0],l[::2,1], alpha=0.5,color=colour[ii])
        axs[0].set_xscale("log")
        axs[0].set_yscale("linear")
        #axs[1].hist(np.log10(l[:,0]), 50,alpha=0.5,color=colour[ii])
        #plotnum.append(l[np.random.randint(l.shape[0]),2]) # gets a random thing to plot
        #axs[1].set_xscale("log")
        axs[1].plot(model.cluster_centers_[i].ravel(),color=colour[ii],linewidth=linefont)
        x.append(len(l[:,0]))
        ii += 1
    i += 1

axs[1].legend(x,loc='lower right',title="Number of\nsimulations",fontsize=fontsize,title_fontsize=fontsize)
axs[0].set_xlabel("$k^0$",fontsize=fontsize)
st = r'$\alpha$'
axs[0].set_ylabel(st,fontsize=fontsize)
#axs[1].set_xlabel("$Log_{10}(k^0)$",fontsize=fontsize)
#axs[1].set_ylabel("Frequency",fontsize=fontsize)
axs[1].set_xlabel("Datapoint",fontsize=fontsize)
axs[1].set_ylabel("Timeseries Averaged Currents",fontsize=fontsize)

fig.suptitle("$E_{Surf}$",fontsize=fontsize*2)
fig.savefig("ESurfparas.png")
plt.close()

try:
    connection = psycopg2.connect(user=dname[0],
                                      password=dname[1],
                                      host=dname[2],
                                      port=dname[3],
                                      database=dname[4])
    i =1
    for num in plotnum:
        print(num)
        cursor = connection.cursor()
        para = cursor.execute("""SELECT "TotalCurrent" FROM "HarmTab" WHERE "Reaction_ID" = %s""", (int(num),))
        for table in cursor.fetchall():
            curr = table[0]
        print(len(curr))
        fsweep = np.linspace(0, 100, num=int(len(curr) / 2))
        bsweep = np.linspace(100, 0, num=int(len(curr) / 2))
        Rx = np.concatenate((fsweep,bsweep))
        plt.figure()
        plt.plot(Rx,curr)
        #plt.savefig("samplereact/ECat_sample"+str(int(num))+"C"+str(i)+".png")
        i += 1

    print(para)
    # sqlcom = """INSERT INTO time(test,"Ctime") VALUES( %()s, CURRENT_TIMESTAMP )"""

except (Exception, psycopg2.Error) as error:
    print("error,", error)
finally:
        # This is needed for the
    if (connection):
        cursor.close()
        connection.close()

#########################PLOTS THE E systems #############################################

try:
    connection = psycopg2.connect(user=dname[0],
                                      password=dname[1],
                                      host=dname[2],
                                      port=dname[3],
                                      database=dname[4])

    cursor = connection.cursor()
    para = cursor.execute("""SELECT "Reaction_ID", "ksreal", "Ru" FROM "Simulatedparameters" WHERE "ReactionMech" = %s""", ("E",))
    para = []
    for table in cursor.fetchall():
        para.append(table)
    print(para)
    # sqlcom = """INSERT INTO time(test,"Ctime") VALUES( %()s, CURRENT_TIMESTAMP )"""

except (Exception, psycopg2.Error) as error:
    print("error,", error)
finally:
        # This is needed for the
    if (connection):
        cursor.close()
        connection.close()

print(newlines[0],para[0])
newnewline = []
for l in newlines:
    for k in para:
        if l[0] == k[0]:
            newnewline.append([l[0],l[1],[k[1][0],k[2]]])
            break

"""# modification to calc deltaE
for i in range(len(newnewline)):
    newnewline[i][2] = newnewline[i][2][1] - newnewline[i][2][0]"""

#print(newnewline)
ptot = []
for i in range(value):
    p = []
    for l in newnewline:
        if i == l[1]:
            x = l[2]
            x.append(l[0])
            #print(x)
            p.append(x)
    ptot.append(p)

fig, axs = plt.subplots(1, 2, tight_layout=True,figsize=fogsize)
plotnum = []
i = 0
print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
clo = plt.rcParams['axes.prop_cycle'].by_key()['color']
print(clo)
x = []
colour = [clo[1],clo[1],clo[3],clo[2]]
ii = 0
for l in ptot:
    print(len(l))
    if len(l) > 100:
        l = np.array(l)
        print(l.shape)
        axs[0].scatter(l[::2,0],l[::2,1], alpha=0.5,color=colour[ii])
        axs[0].set_xscale("log")
        axs[0].set_yscale("log")
        #axs[1].hist(np.log10(l[:,0]), 50,alpha=0.5,color=colour[ii])
        #plotnum.append(l[np.random.randint(l.shape[0]),2]) # gets a random thing to plot
        #axs[1].set_xscale("log")
        axs[1].plot(model.cluster_centers_[i].ravel(),color=colour[ii],linewidth=linefont)
        x.append(len(l[:,0]))
        ii += 1
    i += 1

axs[1].legend(x,loc='lower right',title="Number of\nsimulations",fontsize=fontsize,title_fontsize=fontsize)
axs[0].set_xlabel("$k^0$",fontsize=fontsize)
st = '$R_u$'
axs[0].set_ylabel(st,fontsize=fontsize)
#axs[1].set_xlabel("$Log_{10}(k^0)$",fontsize=fontsize)
#axs[1].set_ylabel("Frequency",fontsize=fontsize)
axs[1].set_xlabel("Datapoint",fontsize=fontsize)
axs[1].set_ylabel("Timeseries Averaged Currents",fontsize=fontsize)

fig.suptitle("$E$",fontsize=fontsize*2)
fig.savefig("Eparas.png")
plt.close()

############################################# THIS IS FOR THE ECE ################################################
try:
    connection = psycopg2.connect(user=dname[0],
                                      password=dname[1],
                                      host=dname[2],
                                      port=dname[3],
                                      database=dname[4])

    cursor = connection.cursor()
    para = cursor.execute("""SELECT "Reaction_ID", "formalE","Ru" FROM "Simulatedparameters" WHERE "ReactionMech" = %s""", ("ECE",))
    para = []
    for table in cursor.fetchall():
        para.append(table)
    print(para)
    # sqlcom = """INSERT INTO time(test,"Ctime") VALUES( %()s, CURRENT_TIMESTAMP )"""

except (Exception, psycopg2.Error) as error:
    print("error,", error)
finally:
        # This is needed for the
    if (connection):
        cursor.close()
        connection.close()

print(newlines[0],para[0])
newnewline = []
for l in newlines:
    for k in para:
        if l[0] == k[0]:
            newnewline.append([l[0],l[1],k[1],k[2]])
            break


# modification to calc deltaE
for i in range(len(newnewline)):
    newnewline[i][2] = newnewline[i][2][2] - newnewline[i][2][0]
    newnewline[i][3] = newnewline[i][3]#[1]# + newnewline[i][3][0])*0.5

#print(newnewline)
ptot = []
for i in range(value):
    p = []
    for l in newnewline:
        if i == l[1]:
            p.append([l[2],l[3]])
    ptot.append(p)

x = []
fig, axs = plt.subplots(1, 2, tight_layout=True,figsize=fogsize)
i = 0
for l in ptot:
    print(len(l))
    if len(l) > 100:
        ll = np.array(l)
        print(ll.shape)
        #axs[0].scatter(ll[:, 0], ll[:, 1], alpha=0.5)
        #axs[0].set_yscale("log")
        axs[0].hist(ll[:, 0],50,alpha=0.5)
        axs[1].plot(model.cluster_centers_[i].ravel(),linewidth=linefont)
        x.append(len(l))
    i += 1
axs[1].legend(x,loc='lower right',title="Number of\nsimulations",fontsize=fontsize,title_fontsize=fontsize)
axs[0].set_xlabel("$\Delta E$",fontsize=fontsize)
axs[0].set_ylabel("Frequency",fontsize=fontsize)
axs[1].set_xlabel("Datapoint",fontsize=fontsize)
axs[1].set_ylabel("Timeseries Averaged Currents",fontsize=fontsize)

fig.suptitle("$ECE$",fontsize=fontsize*2)
fig.savefig("deltaECE.png")


try:
    connection = psycopg2.connect(user=dname[0],
                                      password=dname[1],
                                      host=dname[2],
                                      port=dname[3],
                                      database=dname[4])
    i =1
    for num in plotnum:
        print(num)
        cursor = connection.cursor()
        para = cursor.execute("""SELECT "TotalCurrent" FROM "HarmTab" WHERE "Reaction_ID" = %s""", (int(num),))
        for table in cursor.fetchall():
            curr = table[0]
        print(len(curr))
        fsweep = np.linspace(0, 100, num=int(len(curr) / 2))
        bsweep = np.linspace(100, 0, num=int(len(curr) / 2))
        Rx = np.concatenate((fsweep,bsweep))
        plt.figure()
        plt.plot(Rx,curr)
        #plt.savefig("samplereact/ECat_sample"+str(int(num))+"C"+str(i)+".png")
        i += 1

    print(para)
    # sqlcom = """INSERT INTO time(test,"Ctime") VALUES( %()s, CURRENT_TIMESTAMP )"""

except (Exception, psycopg2.Error) as error:
    print("error,", error)
finally:
        # This is needed for the
    if (connection):
        cursor.close()
        connection.close()


