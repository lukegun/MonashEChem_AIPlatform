import psycopg2
import scipy as sci
from scipy.special import jv, yn    # jv = first order, yn =second kind
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance
from scipy.spatial.distance import cdist
from tslearn import metrics

# dtwsak64
modelloc = "DCclustersettings_dtw_saiko64/trainedtimeseries-1.hdf5"
N = 10423 # reaction ID
modellabel = 2
metric = "dtwsak64"

# softdtw
modelloc = "DCclustersettings_softdtw/trainedtimeseries-1.hdf5"
N = 10423 # reaction ID
modellabel = 3
metric = "softdtw"

# dtw
modelloc = "DCclustersettings_dtw_V1/trainedtimeseries-1.hdf5"
N = 10423 # reaction ID
modellabel = 0
metric = "dtw" 


dname = "timeseriestest"#"DC_db_kf"
#metric = "dtw" #softdtw, dtw, euclidean, dtwsak64

model = TimeSeriesKMeans.from_hdf5(path=modelloc)
barycenter = model.cluster_centers_[modellabel].ravel()

#help("modules")
try:
    connection = psycopg2.connect(user = "postgres",
                                  password = "password",
                                  host = "host",
                                  port = "port",
                                  database = dname)

    cursor = connection.cursor()

    #N = 17152 # EC low diff
    #N = 4000
    print("Reaction ID: " + str(N))
    #N = 18446

    #print(connection.get_dsn_parameters(),"\n")
    #"HarmCol6"
    curr = cursor.execute("""SELECT "TotalCurrent" FROM "HarmTab" WHERE "Reaction_ID" = %s""", (N,))

    for table in cursor.fetchall():
        #print(table)
        x = table[0]

    para = cursor.execute("""SELECT * FROM "Simulatedparameters" WHERE "Reaction_ID" =  %s""", (N,))
    for table in cursor.fetchall():
        para = table
    print(para)
    #sqlcom = """INSERT INTO time(test,"Ctime") VALUES( %()s, CURRENT_TIMESTAMP )"""

except (Exception, psycopg2.Error) as error:
    print("error,", error)
finally:
    # This is needed for the
    if(connection):
        cursor.close()
        connection.close()

    print("it did something")
cursor.close()
connection.close()

plt.plot(x)
#plt.plot(tot,x[:int(len(x)/4)])
plt.savefig("fucker.png")
plt.close()

# decimate the system
sz = len(barycenter)
ndeci = int(len(x)/sz)
s_y2 = x[::ndeci]

# converts to correct format
barycenter = np.array(barycenter)
s_y2 = np.array(s_y2)

print("FUCK")
barycenter = barycenter.reshape(-1, 1)
s_y2 = s_y2.reshape(-1, 1)

print(barycenter.shape)
print(s_y2)

s_y2 = TimeSeriesScalerMinMax(value_range=(-1., 1.)).fit_transform([s_y2])
s_y2 = TimeSeriesScalerMeanVariance().fit_transform(s_y2)

s_y2 = s_y2.reshape(-1, 1)#[:,:,0]
print(s_y2)
print("cunt")

if metric == "dtw":
    path, sim = metrics.dtw_path(barycenter, s_y2)
    mat = metrics.cdist_dtw(barycenter, s_y2)
    del sim
elif metric == "softdtw":
    path, sim = metrics.soft_dtw_alignment(barycenter, s_y2)
    #print(path)
    mat = metrics.cdist_soft_dtw(barycenter, s_y2)
elif metric == "dtwsak64":
    path, sim = metrics.dtw_path(barycenter, s_y2,global_constraint= "sakoe_chiba",sakoe_chiba_radius = 64 )
    mat = metrics.cdist_dtw(barycenter, s_y2,global_constraint= "sakoe_chiba",sakoe_chiba_radius = 64)

    del sim

plt.figure(1, figsize=(8, 8))

# definitions for the axes
left, bottom = 0.01, 0.1
w_ts = h_ts = 0.2
left_h = left + w_ts + 0.02
width = height = 0.65
bottom_h = bottom + height + 0.02

rect_s_y = [left, bottom, w_ts, height]
rect_gram = [left_h, bottom, width, height]
rect_s_x = [left_h, bottom_h, width, h_ts]

ax_gram = plt.axes(rect_gram)
ax_s_x = plt.axes(rect_s_x)
ax_s_y = plt.axes(rect_s_y)

if metric != "dtwsak64":
    ax_gram.imshow(mat, origin='lower')
    ax_gram.axis("off")
    ax_gram.autoscale(False)

elif metric == "dtwsak64":

    for i in range(128):
        if i >64:
            mat[i,:-64 + i] = np.nan
        else:
            mat[i,64+i:] = np.nan
    mat = np.ma.masked_where(np.isnan(mat), mat)
    print(mat)
    ax_gram.imshow(mat[::-1,:])
    ax_gram.axis("on")
    ax_gram.autoscale(False)

if metric == "dtw":
    ax_gram.plot([j for (i, j) in path], [i for (i, j) in path], "w-",
             linewidth=3.)
elif metric == "dtwsak64":

    ax_gram.plot([j for (i, j) in path], [127 - i for (i, j) in path], "w-",
                     linewidth=3.)

else:
    ax_gram.imshow(path, origin='lower')

ax_s_x.plot(np.arange(sz), s_y2, "b-", linewidth=3.)
ax_s_x.axis("off")
ax_s_x.set_xlim((0, sz - 1))

ax_s_y.plot(- barycenter, np.arange(sz), "b-", linewidth=3.)
ax_s_y.axis("off")
ax_s_y.set_ylim((0, sz - 1))

plt.tight_layout()
plt.savefig(metric + "fig.png")

