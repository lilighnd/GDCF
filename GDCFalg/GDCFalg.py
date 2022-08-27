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
from NeighbourHex import *
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
from sklearn.metrics.cluster import normalized_mutual_info_score
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import getopt, sys
import pandas as pd
import os
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.cluster import DBSCAN
# from numpyencoder import NumpyEncoder
# --------------------------------------save code--------------------------------------------------
'''countg=0
grid=0
a = None
Neighbour = None
forest = None
cluster_num = 0
gprim=0
Forest=[]
Alltime=0
r1=0
rdb=0
s_obj = {
        
        "count_g" : countg,
        "neighbour" : Neighbour,
        "a" : a,
        "forest" : forest,
        "cluster_number" : cluster_num,
        "count_gprim" : gprim,
        "forest" : Forest,
        "alltime" : Alltime,
        "rand_index" : r1,
        "rand_indexdb" : rdb

}
save_object = json.dumps(s_obj)

with open("/content/drive/MyDrive/Colab Notebooks/saveobject.json", "w") as outfile:
    outfile.write(save_object)'''

#------------------------------------------------------------------------------------------
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
Noise = 0
Random_state = 42
features = 2
Centers = 3
Epsilon = 0.07
Minpoints = 3
argumentList = sys.argv[1:]
i = 0
SortGrids = "LDF"
# Options
options = "d:M:n:N:r:f:c:e:m:i:s:"
 
# Long options
long_options = ["Dataset", "Mode", "Number_Data", "Noise", "Random_state", "features", "Centers", "Epsilon", "Minpoints","i","SortGrids"]

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

    elif currentArgument in ("-i", "--i"):
        i = currentValue

    elif currentArgument in ("-s", "--SortGrids"):
        SortGrids = currentValue

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
    "i" : i,
    "sort_grids" : SortGrids
}

print(f"mode : {Mode}")
json_object = json.dumps(Obj, indent = 9)
with open("/content/drive/MyDrive/Colab Notebooks/inputobject.json", "w") as outfile:
    outfile.write(json_object)
# --------------------------------------------------------------------------------------------------
with open('/content/drive/MyDrive/Colab Notebooks/inputobject.json', 'r') as openfile:
  
    # Reading from json file
    json_object = json.load(openfile)







# print(f"mode : {Mode}")
#----------------------moons,blobs,circle----------------------------------
# m = DataSet.data()
# True_label = m[1]
# m = m[0].Data
#----------------------generate random data without label for test---------
# print(f"type(data)1 : {type(m)}")
# m = np.random.RandomState(0).randn( int(json_object["n_samples"]), 2)
# print(f"type(data)2 : {type(m)}")
# print(f"data : {m[0]}")
# print(f"data : {m[1]}")

#------------------------------test data curet-----------------------------
# print("start upload dataset")
# df = pd.DataFrame(pd.read_excel(f'/content/drive/MyDrive/Colab Notebooks/datawithnoise.xlsx'))
# print(df)
# print(type(df))
# records = df.to_numpy()
# result = list(records)
# True_label=[]
# m=[]
# for i in range(len(result)):
#     True_label.append(result[i][2])
#     m.append([result[i][0],result[i][1]])

# print(f"m :{m}")
# print(True_label)
# Data=np.transpose(np.array(m))#type of Data is array and Data is transpose of m
# print(f"Data :{Data}")


# Experiment 1

# for n in [ 2000,  5000 ]:
#     Evaluate(n)

# def Evaluate(n):
#     db = CreateDatabase(n)
#     data = datasets.make_moons(n_samples=n,noise=Noise,random_state=R)
#     result =  RunMethod(db)  # recall , precisiion, ...
#     SaveResult(result)

# file
# n =20000, precision = 0.34 , recall = 0.99, ...

# Experiment 1





dim = int(json_object["features"])
m = DataSet.data()
True_label = m[1]
m = m[0].Data#type of m and True_labels is List
Data=np.transpose(np.array(m))#type of Data is array and Data is transpose of m
# print(m)
clustering = DBSCAN(eps= float(json_object["Eps"]), min_samples=int(json_object["Minpts"])).fit(m)
R1 = adjusted_rand_score(True_label, clustering.labels_)
print(R1)
#-----------------------------------start iris---------------------------------------------
# m=[[5.1,3.5,1.4,0.2],
# [4.9,3,1.4,0.2],
# [4.7,3.2,1.3,0.2],
# [4.6,3.1,1.5,0.2],
# [5,3.6,1.4,0.2],
# [5.4,3.9,1.7,0.4],
# [4.6,3.4,1.4,0.3],
# [5,3.4,1.5,0.2],
# [4.4,2.9,1.4,0.2],
# [4.9,3.1,1.5,0.1],
# [5.4,3.7,1.5,0.2],
# [4.8,3.4,1.6,0.2],
# [4.8,3,1.4,0.1],
# [4.3,3,1.1,0.1],
# [5.8,4,1.2,0.2],
# [5.7,4.4,1.5,0.4],
# [5.4,3.9,1.3,0.4],
# [5.1,3.5,1.4,0.3],
# [5.7,3.8,1.7,0.3],
# [5.1,3.8,1.5,0.3],
# [5.4,3.4,1.7,0.2],
# [5.1,3.7,1.5,0.4],
# [4.6,3.6,1,0.2],
# [5.1,3.3,1.7,0.5],
# [4.8,3.4,1.9,0.2],
# [5,3,1.6,0.2],
# [5,3.4,1.6,0.4],
# [5.2,3.5,1.5,0.2],
# [5.2,3.4,1.4,0.2],
# [4.7,3.2,1.6,0.2],
# [4.8,3.1,1.6,0.2],
# [5.4,3.4,1.5,0.4],
# [5.2,4.1,1.5,0.1],
# [5.5,4.2,1.4,0.2],
# [4.9,3.1,1.5,0.1],
# [5,3.2,1.2,0.2],
# [5.5,3.5,1.3,0.2],
# [4.9,3.1,1.5,0.1],
# [4.4,3,1.3,0.2],
# [5.1,3.4,1.5,0.2],
# [5,3.5,1.3,0.3],
# [4.5,2.3,1.3,0.3],
# [4.4,3.2,1.3,0.2],
# [5,3.5,1.6,0.6],
# [5.1,3.8,1.9,0.4],
# [4.8,3,1.4,0.3],
# [5.1,3.8,1.6,0.2],
# [4.6,3.2,1.4,0.2],
# [5.3,3.7,1.5,0.2],
# [5,3.3,1.4,0.2],
# [7,3.2,4.7,1.4],
# [6.4,3.2,4.5,1.5],
# [6.9,3.1,4.9,1.5],
# [5.5,2.3,4,1.3],
# [6.5,2.8,4.6,1.5],
# [5.7,2.8,4.5,1.3],
# [6.3,3.3,4.7,1.6],
# [4.9,2.4,3.3,1],
# [6.6,2.9,4.6,1.3],
# [5.2,2.7,3.9,1.4],
# [5,2,3.5,1],
# [5.9,3,4.2,1.5],
# [6,2.2,4,1],
# [6.1,2.9,4.7,1.4],
# [5.6,2.9,3.6,1.3],
# [6.7,3.1,4.4,1.4],
# [5.6,3,4.5,1.5],
# [5.8,2.7,4.1,1],
# [6.2,2.2,4.5,1.5],
# [5.6,2.5,3.9,1.1],
# [5.9,3.2,4.8,1.8],
# [6.1,2.8,4,1.3],
# [6.3,2.5,4.9,1.5],
# [6.1,2.8,4.7,1.2],
# [6.4,2.9,4.3,1.3],
# [6.6,3,4.4,1.4],
# [6.8,2.8,4.8,1.4],
# [6.7,3,5,1.7],
# [6,2.9,4.5,1.5],
# [5.7,2.6,3.5,1],
# [5.5,2.4,3.8,1.1],
# [5.5,2.4,3.7,1],
# [5.8,2.7,3.9,1.2],
# [6,2.7,5.1,1.6],
# [5.4,3,4.5,1.5],
# [6,3.4,4.5,1.6],
# [6.7,3.1,4.7,1.5],
# [6.3,2.3,4.4,1.3],
# [5.6,3,4.1,1.3],
# [5.5,2.5,4,1.3],
# [5.5,2.6,4.4,1.2],
# [6.1,3,4.6,1.4],
# [5.8,2.6,4,1.2],
# [5,2.3,3.3,1],
# [5.6,2.7,4.2,1.3],
# [5.7,3,4.2,1.2],
# [5.7,2.9,4.2,1.3],
# [6.2,2.9,4.3,1.3],
# [5.1,2.5,3,1.1],
# [5.7,2.8,4.1,1.3],
# [6.3,3.3,6,2.5],
# [5.8,2.7,5.1,1.9],
# [7.1,3,5.9,2.1],
# [6.3,2.9,5.6,1.8],
# [6.5,3,5.8,2.2],
# [7.6,3,6.6,2.1],
# [4.9,2.5,4.5,1.7],
# [7.3,2.9,6.3,1.8],
# [6.7,2.5,5.8,1.8],
# [7.2,3.6,6.1,2.5],
# [6.5,3.2,5.1,2],
# [6.4,2.7,5.3,1.9],
# [6.8,3,5.5,2.1],
# [5.7,2.5,5,2],
# [5.8,2.8,5.1,2.4],
# [6.4,3.2,5.3,2.3],
# [6.5,3,5.5,1.8],
# [7.7,3.8,6.7,2.2],
# [7.7,2.6,6.9,2.3],
# [6,2.2,5,1.5],
# [6.9,3.2,5.7,2.3],
# [5.6,2.8,4.9,2],
# [7.7,2.8,6.7,2],
# [6.3,2.7,4.9,1.8],
# [6.7,3.3,5.7,2.1],
# [7.2,3.2,6,1.8],
# [6.2,2.8,4.8,1.8],
# [6.1,3,4.9,1.8],
# [6.4,2.8,5.6,2.1],
# [7.2,3,5.8,1.6],
# [7.4,2.8,6.1,1.9],
# [7.9,3.8,6.4,2],
# [6.4,2.8,5.6,2.2],
# [6.3,2.8,5.1,1.5],
# [6.1,2.6,5.6,1.4],
# [7.7,3,6.1,2.3],
# [6.3,3.4,5.6,2.4],
# [6.4,3.1,5.5,1.8],
# [6,3,4.8,1.8],
# [6.9,3.1,5.4,2.1],
# [6.7,3.1,5.6,2.4],
# [6.9,3.1,5.1,2.3],
# [5.8,2.7,5.1,1.9],
# [6.8,3.2,5.9,2.3],
# [6.7,3.3,5.7,2.5],
# [6.7,3,5.2,2.3],
# [6.3,2.5,5,1.9],
# [6.5,3,5.2,2],
# [6.2,3.4,5.4,2.3],
# [5.9,3,5.1,1.8]]
# Data=np.transpose(np.array(m))#type of Data is array and Data is transpose of m
#------------------------------------------Normalized----------------------------------------------------
# m = NormalizeData(m)#normalize
# Data=np.transpose(np.array(m))#type of Data is array and Data is transpose of m
# def NormalizeData(data):
#     return (data - np.min(data)) / (np.max(data) - np.min(data))
#--------------------------------------------------------------------------------------------------------

#---------------------------------LL---------------------------------------------------------------------

# t1=time.time()
# x = LocallyLinearEmbedding(n_components=2)
# X_transformed = x.fit_transform(m[:150])
# t2=time.time()-t1
# print(f"Data : {X_transformed}")
# print(f"t-LLE : {t2}")

# m=X_transformed
# Data=np.transpose(np.array(m))#type of Data is array and Data is transpose of m
# dim=2
#--------------------------------------------------------------------------------------------------------
# print(f"data : {m}")
# True_label=[
# 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
# ,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
# 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
#------------------------------------end iris--------------------------------------------------------------------
# m = DataSet.data()
# True_label = m[1]
# m = m[0].Data#type of m and True_labels is List
# Data=np.transpose(np.array(m))#type of Data is array and Data is transpose of m
# ------Read Data another way---------
# m = DataSet.dataclutot()
# True_label = m[1]
# m = m[0].Data#type of m and True_labels is List
# Data=np.transpose(np.array(m))

# True_label = moons[1]
# m=moons[0]




# np.savetxt('/content/drive/MyDrive/Colab Notebooks/data.csv',Data,delimiter=',')
# print(f"Data : {Data,len(Data),len(Data[0])}")
# print("load Data")


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
# 




# print(Data,True_label)
Eps = float(json_object["Eps"])
MinPts = int(json_object["Minpts"])
print(f"dim :{dim}")
G = []
start_time = time.time()
# ---------------Hex Grids-----------
modeGrid = int(json_object["mode_grid"])
# Hexagonal
start_time_grid = time.time()
print(modeGrid,int(1))
if modeGrid == int(1):
    print("Hex mode")
    parts=make_Hex(Data,Eps)
    Grids,gridData,numGrid_dim=parts.GridHex()

    

    # with open('/content/drive/MyDrive/Colab Notebooks/saveobject.json', 'r') as openfile:
  
    #     # Reading from json file
    #     save_object = json.load(openfile)

    # s_obj["datagrid"] = gridData
    # save_object = json.dumps(s_obj)
    # with open("/content/drive/MyDrive/Colab Notebooks/saveobject.json", "w") as outfile:
    #         outfile.write(save_object)
    

    print("run grid")


# -------------Square Grids----------
if modeGrid == int(2):
    print("square mode")
    parts = Make_Square(Data,Eps,dim)
    # print(Data)
    # Grids, gridData = parts.GridHex()
    Grids, gridData,numGrid_dim = parts.GridSqn()
    # Grids, gridData,numGrid_dim = parts.GridHex()
    print("run grid")
print(f"time_grid = {time.time() - start_time_grid}")
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
# print(f"core:{CoreGrid,CoreObject}")
print("run core")


# for i in range(len(gridData)):
#    if len(gridData[i])>=MinPts:
#        G.append(Grids[i])
#    else:
#        G.append([])


# ----------------------------------------HGB--------------------------------------------------------------
start_time_hgb=time.time()
HGBmatrix = HGB(Grids,dim,numGrid_dim)
B = HGBmatrix.BuildHGB()
print("run HGB")
print(f"time_hgb = {time.time() - start_time_hgb}")

# _______________________________________GDCF________________________________________________________________-
start_time_gdcf=time.time()
gdcf = GDCF(CoreGrid, CoreObject, dim, B, MinPts, Eps)
if modeGrid == int(1):
    ClusterForest = gdcf.BuildGDCF(json_object["sort_grids"],"Hex", gridData, m, Grids)
if modeGrid == int(2):
    ClusterForest = gdcf.BuildGDCF(json_object["sort_grids"],"Square", gridData, m, Grids)
print("run GDCF")
print(f"time_gdcf = {time.time() - start_time_gdcf}")


Pred_label = []
for i in range(len(ClusterForest)):
    Pred_label.append(ClusterForest[i][-1])
# print(f"pred:{Pred_label}")   
alltime = time.time() - start_time
R1 = adjusted_rand_score(True_label, Pred_label)
f1 = metrics.f1_score(True_label, Pred_label,average='micro')
nmi = normalized_mutual_info_score(True_label, Pred_label)
print(f"R1,alltime : {R1,nmi,f1,alltime}")

excel_name_label = f'/content/drive/MyDrive/Colab Notebooks/mylabels.xlsx'
df_plabels = pd.DataFrame(Pred_label)
df_plabels.to_excel(excel_name_label,index=False)
print("Labels saved")

# f2 = metrics.f1_score(True_label, db.labels_, average='weighted')

#----------------------------------------file for find best Eps,cols=90;rows=eps,R1------------------------------------------
namefile = str(json_object["data"]) + str(json_object["n_samples"]) + str(json_object["sort_grids"]) + str(json_object["mode_grid"])

ls = []
ls.append(json_object["Eps"])
ls.append(R1)
ls.append(alltime)
df = pd.DataFrame(ls) 

excel_name = f'/content/drive/MyDrive/Colab Notebooks/{namefile}"R1"agg.xlsx'
print(excel_name)
df_source = None
if os.path.exists(excel_name):
    print("os.path.exists(excel_name)")
    df_source = pd.DataFrame(pd.read_excel(excel_name))
    print("os if is ok")

if df_source is not None:
    print("df_source is not None")
    df_source[json_object["i"]]=ls
    df_dest = df_source
    print("df_source if is ok")

else:
    print("not exist")
    df_dest = df
    print("ok if not exist")


df_dest.to_excel(excel_name,index=False)
print("Save excel")

#----------------------------------------file for presentaion results------------------------------------------------------------------
cols=['DataSize','Mode_Grid','Time','Improvment','SortWay','DataSetType','ARI','F1-Score','Purity','Precision']
df = pd.DataFrame(columns=cols)

# ls = []
# ls.append(json_object["Eps"])
# ls.append(f1)
# ls.append(alltime)
# df = pd.DataFrame(ls) 
excel_name = f'/content/drive/MyDrive/Colab Notebooks/resultfile.xlsx'
print(excel_name)
df_source = None
if os.path.exists(excel_name):
    print("os.path.exists(excel_name)")
    df_source = pd.DataFrame(pd.read_excel(excel_name))
    print("os if is ok")

if df_source is not None:
    print("df_source is not None")
    df_source.at[json_object["i"],'DataSize'] = json_object["n_samples"]
    df_source.at[json_object["i"],'Mode_Grid'] = json_object["mode_grid"]
    df_source.at[json_object["i"],'Time'] = alltime
    # if int(json_object["mode_grid"]) == 2:
    #     print("sqsqsq")
    #     df_source.at[int(json_object["i"])-1,'Improvment'] = (df_source.iloc[int(json_object["i"])]['Time']-df_source.iloc[int(json_object["i"]-1)]['Time'])/df_source.iloc[int(json_object["i"])]['Time']
    df_source.at[json_object["i"],'SortWay'] = json_object["sort_grids"]
    df_source.at[json_object["i"],'DataSetType'] = json_object["data"]
    df_source.at[json_object["i"],'ARI'] = R1
    df_dest = df_source
    print("df_source if is ok")

else:
    print("not exist")
    df_dest = df
    print("ok if not exist")


df_dest.to_excel(excel_name,index=False)
print("Save excel")
# db = DBSCAN(eps=Eps, min_samples=MinPts).fit(m)
# db.labels_ = list(np.float_(db.labels_))



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
# print(True_label)
# print(Pred_label)
# print(db.labels_)
print("End Run")

# R1 = adjusted_rand_score(True_label, Pred_label)
# R2 = adjusted_rand_score(True_label, db.labels_)

# print(R1, R2, f"alltime:{alltime}")

# M1 = adjusted_mutual_info_score(True_label, Pred_label)
# M2 = adjusted_mutual_info_score(True_label, db.labels_)

# Calinski_Harabasz1 = calinski_harabasz_score(m, Pred_label)
# Calinski_Harabasz1 = calinski_harabasz_score(m, db.labels_)

# FM1 = metrics.fowlkes_mallows_score(True_label, Pred_label)
# FM1 = metrics.fowlkes_mallows_score(True_label, db.labels_)

# S1 = metrics.silhouette_score(m, Pred_label)
# S2 = metrics.silhouette_score(m, db.labels_)

# ss1 = metrics.silhouette_samples(m, Pred_label)
# ss2 = metrics.silhouette_samples(m, db.labels_)

# db1 = metrics.davies_bouldin_score(m, Pred_label)
# db2 = metrics.davies_bouldin_score(m, db.labels_)


# per1 = metrics.precision_score(True_label, Pred_label, average='weighted')
# per2 = metrics.precision_score(
#     True_label, db.labels_, average='weighted')  # error


# v1 = metrics.v_measure_score(True_label, Pred_label)
# v1 = metrics.v_measure_score(True_label, db.labels_)

# pu = Evaluation(True_label, Pred_label)
# pu1 = pu.purity_score()
# p = Evaluation(True_label, db.labels_)
# pu2 = p.purity_score()


# f1 = metrics.f1_score(True_label, Pred_label, average='weighted')
# f2 = metrics.f1_score(True_label, db.labels_, average='weighted')
# alltime = time.time() - start_time
# # p=Plot(D,0)


# print(alltime)
