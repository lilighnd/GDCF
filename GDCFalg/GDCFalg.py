from HGB import *
from CoreGrids import *
from Evaluation import *
from Grid import *
from GDCF import *
from Kdist import *
from Plot import *
from DataSet import *
from make_Hex import *
from Make_Square import *
from NeighbourGridQuery import *
import time
import threading
from sklearn.datasets import make_circles
from sklearn.metrics import pairwise
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
#from sklearn.metrics import rand_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import r2_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import calinski_harabasz_score
from sklearn import metrics


import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import getopt, sys
import json
# --------------------------------------read data--------------------------------------------------
Grids=[]
gridData=[]
dists=[]
corepoints=[]
coregrids=[]
grid=0
hgb=[]
g=[]
gprim=[]
Forest=[]
Alltime=0
r1=0
rdb=0
s_obj = {
        "grid" : Grids,
        "datagrid" : gridData,
        "dintances" : dists,
        "coreobjects" : corepoints,
        "coregrids" : coregrids,
        "count_grid" : grid,
        "hgbmatrix" : hgb,
        "count_g" : g,
        "count_gprim" : gprim,
        "forest" : Forest,
        "alltime" : Alltime,
        "rand_index" : r1,
        "rand_indexdb" : rdb

}
# n_samples =1000000


#noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)

# moons = datasets.make_moons (n_samples=n_samples, shuffle=False, noise=0.05, random_state=None)

# blobs = datasets.make_blobs(n_samples=n_samples, n_features = 2, 
#                   centers = 3,cluster_std = 0,random_state=42)

#no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
#random_state = 170
#X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
#transformation = [[0.6, -0.6], [-0.4, 0.8]]
#X_aniso = np.dot(X, transformation)
#aniso = (X_aniso, y)

# blobs with varied variances
# varied = datasets.make_blobs(
#    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
# )
Dataset = "blob"
Mode = 2
Number_Data = 38000
Noise = 0.05
Random_state = 42
features = 2
Centers = 3
Epsilon = 0.07
Minpoints = 3
argumentList = sys.argv[1:]
# Options
options = "d:M:n:N:r:f:c:e:m:"
 
# Long options
long_options = ["Dataset", "Mode", "Number_Data", "Noise", "Random_state", "features", "Centers", "Epsilon", "Minpoints"]

arguments, values = getopt.getopt(argumentList, options, long_options)

for currentArgument, currentValue in arguments:

    if currentArgument in ("-d", "--Dataset"):
        Dataset = currentValue


    elif currentArgument in ("-M", "--Mode"):
        Mode = currentValue
            

    elif currentArgument in ("-n", "--Number_Data"):
        Number_Data = currentValue

    elif currentArgument in ("-N", "--Noise"):
        Noise = currentValue

    elif currentArgument in ("-r", "--Random_state"):
        Random_state = currentValue
            
    elif currentArgument in ("-f", "--features"):
        features = currentValue

    elif currentArgument in ("-c", "--Centers"):
        Centers = currentValue

    elif currentArgument in ("-e", "--Epsilon"):
        Epsilon = currentValue
    
    elif currentArgument in ("-m", "--Minpoints"):
        Minpoints = currentValue

Obj = {

    "data" : Dataset,
    "mode_grid" : Mode,
    "n_samples" : Number_Data,
    "noise" : Noise,
    "random_state" : Random_state,
    "features" : features,
    "centers" : Centers,
    "Eps" : Epsilon,
    "Minpts" : Minpoints,
}

print(f"mode : {Mode}")
json_object = json.dumps(Obj, indent = 9)
with open("/content/drive/MyDrive/Colab Notebooks/inputobject.json", "w") as outfile:
    outfile.write(json_object)
# --------------------------------------------------------------------------------------------------
with open('/content/drive/MyDrive/Colab Notebooks/inputobject.json', 'r') as openfile:
  
    # Reading from json file
    json_object = json.load(openfile)
  
print(f"mode : {Mode}")

m = DataSet.data()
True_label = m[1]
m = m[0].Data



# db = DBSCAN(eps=0.07, min_samples=5).fit(m)
# db.labels_ = list(np.float_(db.labels_))
# R2=adjusted_rand_score(True_label, db.labels_)
# print(R2)
# print(True_label)
# print(db.labels_)
# print("end db")



# print("load m")
# ------Read Data another way---------
#path = f'..\\GDCFalg\\blobsData.csv'
#df = pd.read_csv(path)
#blobs = df.values.tolist()
# m=blobs
#path = f'..\\GDCFalg\\blobsDataLabels.csv'
#df = pd.read_csv(path)
#True_label = df.values.tolist()


# True_label = moons[1]
# m=moons[0]

Data= [[] for i in range(len(m[0]))]
for dim in range(len(m[0])):
    for i in range(len(m)):
        Data[dim].append(m[i][dim])


# print(f"Data : {Data,len(Data),len(Data[0])}")
# print("load Data")

# minx=-(min(Data[0]))
# miny=-(min(Data[1]))




#for i in range(len(Data[0])):Data[0][i]=Data[0][i]
#for j in range(len(Data[1])):Data[1][j]=Data[1][j]

#for i in range(len(m)):m[i][0]=m[i][0]
#for j in range(len(m)):m[j][1]=m[j][1]


# ---------for first preprocess dataset-----
# Data=blobs
# D=[[] for i in range(len(Data[0][0]))]
# for dim in range(len(Data[0][0])):
#    for data in Data[0]:
#        D[dim].append(data[dim])

# np.savetxt("/content/drive/MyDrive/Colab Notebooks/moonsData38.csv", 
#           m,
#           delimiter =",", 
#           fmt ='% s')

# np.savetxt("/content/drive/MyDrive/Colab Notebooks/moonsLabels38.csv", 
#           True_label,
#           delimiter =",", 
#           fmt ='% s')
# print("Save Data")
# np.savetxt("blobsData150.csv",
#   Data[0],
#   delimiter =",",
#   fmt ='% s')

# np.savetxt("blobsDataLabels150.csv",
#   Data[1],
#   delimiter =",",
#   fmt ='% s')

# ---------------run dbscan--------------
# start_time1 = time.time()
# db = DBSCAN(eps=5, min_samples=5).fit(m)
# # core = db.core_sample_indices_
# R2=adjusted_rand_score(True_label, db.labels_)
# MinPts = 5

# ------------------------------Find the best of Epsilon & MinPoints------------------------------------------
# all=np.zeros([10000,3])
# i=0
# for e in np.arange(0.01,5,0.01):
#    for min in range(2,20):
#        db = DBSCAN(eps=e, min_samples=min).fit(m)
#        R2=adjusted_rand_score(True_label, db.labels_)
#        all[i][0]=R2
#        all[i][1]=e
#        all[i][2]=min
#        i+=1

# indx=np.where(all==max(all[:,0]))
# best=all[indx[0]]
# -------------------------------------End----------------------------------------------------

# --------------------------------------plot data--------------------------------------------
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# # Major ticks every 20, minor ticks every 5
# major_ticks = np.arange(0, 4.5, 0.17)
# minor_ticks = np.arange(0, 4.5, 0.17)
# ax.set_xticks(major_ticks)
# ax.set_xticks(minor_ticks, minor=True)
# ax.set_yticks(major_ticks)
# ax.set_yticks(minor_ticks, minor=True)
# ax.set_xlim([0,5])
# ax.set_ylim([0,5])
# # And a corresponding grid
# ax.grid(which='both')
# plt.grid()
# p=Plot(Data,0)
# p2=p.plotClusters()
# plt.show()

# ------------------------------------------------------------------------------

# ------------------------------Kdist for find Eps-------------------------------------------------------
# kdist=Kdist(Data,3)
# kd=kdist.calculate_kn_distance()
# kdist2=Kdist(Data[1],4)
#eps_dist = kdist.calculate_kn_distance()
# p=plt.hist(eps_dist,bins=30)
# plt.ylabel('n');
#plt.xlabel('Epsilon distance')
# plt.show()
# ------------------------------------------------------------------------------------------------------


# ----------------------------------Kinds of Grid--------------------------------------------------------
# MinPts = 3
# Eps = 0.07
Eps = float(json_object["Eps"])
MinPts = int(json_object["Minpts"])
G = []
start_time = time.time()
# ---------------Hex Grids-----------
modeGrid = int(json_object["mode_grid"])
# Hexagonal
print(modeGrid,int(1))
if modeGrid == int(1):
    print("Hex mode")
    parts=make_Hex(Data,Eps)
    Grids,gridData=parts.GridHex()

#-*-*-*-*-*-*-*-*-Json-*-*-*-*-*-*-*-*-*-*-*-
    s_obj["grid"] = Grids
    s_obj["datagrid"] = gridData

    save_object = json.dumps(s_obj["grid"],indent=13)
    with open("/content/drive/MyDrive/Colab Notebooks/saveobject.json", "w") as outfile:
        outfile.write(save_object)

    save_object = json.dumps(s_obj["datagrid"],indent=13)
    with open("/content/drive/MyDrive/Colab Notebooks/saveobject.json", "w") as outfile:
        outfile.write(save_object)

    print("run grid")


# -------------Square Grids----------
if modeGrid == int(2):
    print("square mode")
    parts = Make_Square(Data,Eps)
    Grids, gridData = parts.GridHex()
    print("run grid")

# -----------------------Recalling Saved Cores-------------------
#path = '/content/drive/MyDrive/Colab Notebooks/CoreGrids.csv'
#df = pd.read_csv(path)
#CoreGrid = df.values.tolist()

#path = '/content/drive/MyDrive/Colab Notebooks/CoreObjects.csv'
#df = pd.read_csv(path)
#CoreObject = df.values.tolist()

#path = f'..\\GDCFalg\\CoreGrids.csv'
#df = pd.read_csv(path)
#CoreGrid = df.values.tolist()

#path = f'..\\GDCFalg\\CoreObjects.csv'
#df = pd.read_csv(path)
#CoreObject = df.values.tolist()

core = CoreGrids(Grids, gridData, Data, Eps, MinPts, m)
CoreGrid, CoreObject = core.Find_CoreGrids()
print("run core")


# for i in range(len(gridData)):
#    if len(gridData[i])>=MinPts:
#        G.append(Grids[i])
#    else:
#        G.append([])


# ----------------------------------------HGB--------------------------------------------------------------
HGBmatrix = HGB(Grids, 2)
B = HGBmatrix.BuildHGB()
print("run HGB")

# _______________________________________GDCF________________________________________________________________-

gdcf = GDCF(CoreGrid, CoreObject, 2, B, MinPts, Eps)
ClusterForest = gdcf.BuildGDCF("LDF", gridData, m, Grids)
print("run GDCF")


Pred_label = []
for i in range(len(ClusterForest)):
    Pred_label.append(ClusterForest[i][-1])
    
alltime = time.time() - start_time
print(alltime)



db = DBSCAN(eps=Eps, min_samples=MinPts).fit(m)
db.labels_ = list(np.float_(db.labels_))
#plt.subplot(1, 3, 1)
# Getting unique labels
# label=list(map(int,Pred_label))
#u_labels = np.unique(label)
# m=np.array(m)
# plotting the results:

# for i in u_labels:
#    plt.scatter(m[label == i,0] , m[label == i,1 ] , label = i)
# plt.legend()
# plt.title("Grid")

#plt.subplot(1, 3, 2)
# label=list(map(int,db.labels_))
#u_labels = np.unique(label)
# m=np.array(m)
# plotting the results:

# for i in u_labels:
#    plt.scatter(m[label == i,0] , m[label == i,1 ] , label = i)
# plt.legend()
# plt.title("DBSCAN")

#plt.subplot(1, 3, 3)
# label=list(map(int,True_label))
#u_labels = np.unique(label)
# m=np.array(m)
# plotting the results:

# for i in u_labels:
#    plt.scatter(m[label == i,0] , m[label == i,1 ] , label = i)
# plt.legend()
#plt.title("True Label")

# plt.show()


# --------------------------------------Evaluation-----------------------------------------------------------
print("End Run")
alltime = time.time() - start_time

R1 = adjusted_rand_score(True_label, Pred_label)
R2 = adjusted_rand_score(True_label, db.labels_)

print(True_label, Pred_label, db.labels_)
print(R1, R2, alltime)

M1 = adjusted_mutual_info_score(True_label, Pred_label)
M2 = adjusted_mutual_info_score(True_label, db.labels_)

Calinski_Harabasz1 = calinski_harabasz_score(m, Pred_label)
Calinski_Harabasz1 = calinski_harabasz_score(m, db.labels_)

FM1 = metrics.fowlkes_mallows_score(True_label, Pred_label)
FM1 = metrics.fowlkes_mallows_score(True_label, db.labels_)

S1 = metrics.silhouette_score(m, Pred_label)
S2 = metrics.silhouette_score(m, db.labels_)

ss1 = metrics.silhouette_samples(m, Pred_label)
ss2 = metrics.silhouette_samples(m, db.labels_)

db1 = metrics.davies_bouldin_score(m, Pred_label)
db2 = metrics.davies_bouldin_score(m, db.labels_)


per1 = metrics.precision_score(True_label, Pred_label, average='weighted')
per2 = metrics.precision_score(
    True_label, db.labels_, average='weighted')  # error


v1 = metrics.v_measure_score(True_label, Pred_label)
v1 = metrics.v_measure_score(True_label, db.labels_)

pu = Evaluation(True_label, Pred_label)
pu1 = pu.purity_score()
p = Evaluation(True_label, db.labels_)
pu2 = p.purity_score()


f1 = metrics.f1_score(True_label, Pred_label, average='weighted')
f2 = metrics.f1_score(True_label, db.labels_, average='weighted')
alltime = time.time() - start_time
# p=Plot(D,0)


print(alltime)
