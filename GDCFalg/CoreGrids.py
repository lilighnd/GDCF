import numpy as np
from scipy.spatial import distance
import pandas as pd
from sklearn.cluster import DBSCAN
import json
import time


class CoreGrids:
    def __init__(self, Grids, DataInGrids, data, Epsilon, Minpoints, m):
        self.Grids = Grids
        self.PointsInGrids = DataInGrids
        self.Data = data
        self.Eps = Epsilon
        self.MinPts = Minpoints
        self.m = m
        # print(self.m)

    def distances(self, Point):
        # print(f"m,point : {self.m,[Point]}")
        # dist = distance.cdist([Point], self.m, 'euclidean')
        dist = distance.cdist([Point], self.m, 'chebyshev')
        # dist = distance.cdist([Point], self.m, 'cityblock')
        # dist = distance.cdist([Point], self.m, 'jaccard')
        return dist

    def Find_CoreObject(self):
        # print(self.m)
        start_time_coreobject = time.time()
        Core_Objects = []
        for pointOfData in range(len(self.m)):
            dists = self.distances(self.m[pointOfData])
            if (dists <= self.Eps).sum() >= self.MinPts:
                Core_Objects.append(pointOfData)
                # print(f"pointOfData : {pointOfData}") 
                # print(f"dist : {dists}")
                continue

            # count = 0
            # for SecodPointOfData in range(len(self.m)):
            #     if dists[0][SecodPointOfData] <= self.Eps:
            #         count += 1
            # if count >= self.MinPts:
            #     Core_Objects.append(pointOfData)
        # print(f"core obj : {Core_Objects}")nnn
        # print(f"time_coreobject={time.time() - start_time_coreobject}")nnn
        return Core_Objects

    def Find_CoreGrids(self):
        # np.savetxt("/content/drive/MyDrive/Colab Notebooks/distanceblobs.csv",
        #   dists,
        #   delimiter =",",
        #   fmt ='% s')

        #path = '/content/drive/MyDrive/Colab Notebooks/distanceblobs.csv'
        #df = pd.read_csv(path)
        #dists = df.values.tolist()

        #path = f'..\\GDCFalg\\distanceblobs.csv'
        #df = pd.read_csv(path)
        #dists = df.values.tolist()
        CorePoints = self.Find_CoreObject()
        # print(f"Core points : {CorePoints}")
        # print(f"all Grids : {self.Grids}")

        print("Core grid is running")
        Core_Grids = []
        # Core Grids
        start_time_coregrid = time.time()
        for grid in range(len(self.Grids)):
            NotAdd = True
            if len(self.PointsInGrids[grid]) >= self.MinPts:
                Core_Grids.append(self.Grids[grid])
                NotAdd = False
                continue

            for Point_grid in self.PointsInGrids[grid]:
                if Point_grid in CorePoints:
                    Core_Grids.append(self.Grids[grid])
                    NotAdd = False
                    break

            if NotAdd == True:
                Core_Grids.append([])
                # print("empty core grid")nnn

            # for j in range(len(self.PointsInGrids[i])):
            #     Count = 0
            #     for k in range(len(dists[0])):
            #         if dists[self.PointsInGrids[i][j]][k] <= self.Eps:
            #             Count += 1

            #     if Count >= self.MinPts or self.PointsInGrids[i][j] in Core_Objects:
            #         Core_Grids.append(self.Grids[i])
            #         break
            # if self.Grids[i] not in Core_Grids:
            #     Core_Grids.append([])

        # np.savetxt("/content/drive/MyDrive/Colab Notebooks/CoreGrids.csv",
        #   Core_Grids,
        #   delimiter =",",
        #   fmt ='% s')
        # np.savetxt("/content/drive/MyDrive/Colab Notebooks/CoreObjects.csv",
        #   Core_Objects,
        #   delimiter =",",
        # #   fmt ='% s')
        # print(f"time_coregrid={time.time() - start_time_coregrid}")nnn
        # print(f"coregrid={Core_Grids}")nnn
        return Core_Grids, CorePoints

    # def distances(self, Point):
    #     # "Point" is point that checked all distances for another points

    #     # dist=np.zeros((len(self.Data[0]),len(self.Data[0])))
    #     # dist[0][0]=0
    #     # for i in range(len(self.Data[0])):
    #     #    for j in range(len(self.Data[0])):
    #     #        if j>i:
    #     #            dist[j][i]=dist[i][j]=(distance.euclidean([self.Data[0][i],self.Data[1][i]],[self.Data[0][j],self.Data[1][j]]))
    #     #            #dist[j][i]=dist[i][j]=(distance.euclidean([self.Data[0][i][ for ],[self.Data[0][j],self.Data[1][j]]))
    #     #        if i==j:
    #     #            dist[i][j]=0
    #     #        else:
    #     #            continue
    #     dist = distance.cdist([Point], self.m, 'euclidean')
    #     return dist
