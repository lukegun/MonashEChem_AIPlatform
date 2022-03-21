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
for modelloc in modellocfull:
    arrs = []
    i = 0
    for values in Nc:
        model = TimeSeriesKMeans.from_hdf5(path=modelloc[i]+ "/trainedtimeseries-1.hdf5")

        #for j in range(Nc[i]):

        arrs.append(model.inertia_)
        i += 1

    plt.plot(Nc,arrs)

plt.savefig("Ncaccuracy_inertia.png")