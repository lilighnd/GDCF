import numpy as np
from scipy.spatial import distance
import pandas as pd

class CoreGrids:
    """description of class"""
    def __init__(self,Grids,DataInGrids,data,Epsilon,Minpoints):
        self.Grids=Grids
        self.PointsInGrids=DataInGrids
        self.Data=data
        self.Eps=Epsilon
        self.MinPts=Minpoints
        
    def Find_CoreGrids(self):
        #dists=distances(self)
        #np.savetxt("/content/drive/MyDrive/Colab Notebooks/distanceblobs.csv", 
        #   dists,
        #   delimiter =",", 
        #   fmt ='% s')
        
        path = '/content/drive/MyDrive/Colab Notebooks/CoreGrids.csv'
        df = pd.read_csv(path)
        dists = df.values.tolist()



        #path = f'..\\GDCFalg\\distanceblobs.csv'
        #df = pd.read_csv(path)
        #dists = df.values.tolist()

        Core_Grids = []
        Core_Objects = []

        #Core Points
        for i in range(len(dists)):
            count=0
            for j in range(len(dists[0])):
                if dists[i][j]<=self.Eps:
                    count+=1
            if count>=self.MinPts:
                Core_Objects.append(i)

        #Core Grids
        for i in range(len(self.Grids)):
            if len(self.PointsInGrids[i]) >= self.MinPts:
                Core_Grids.append(self.Grids[i])

            if self.Grids[i] not in Core_Grids:
                for j in range(len(self.PointsInGrids[i])):
                    Count = 0
                    for k in range(len(dists[0])):
                        if dists[self.PointsInGrids[i][j]][k] <= self.Eps:
                            Count += 1

                    if  Count >= self.MinPts or self.PointsInGrids[i][j] in Core_Objects:
                        Core_Grids.append(self.Grids[i])
                        break
            if self.Grids[i] not in Core_Grids:
                Core_Grids.append([])
        



        #np.savetxt("/content/drive/MyDrive/Colab Notebooks/CoreGrids.csv", 
        #   Core_Grids,
        #   delimiter =",", 
        #   fmt ='% s')
        #np.savetxt("/content/drive/MyDrive/Colab Notebooks/CoreObjects.csv", 
        #   Core_Objects,
        #   delimiter =",", 
        #   fmt ='% s')
        return Core_Grids,Core_Objects

def distances(self):
    dist=np.zeros((len(self.Data[0]),len(self.Data[0])))
    dist[0][0]=0
    for i in range(len(self.Data[0])):
        for j in range(len(self.Data[0])):
            if j>i:
                dist[j][i]=dist[i][j]=(distance.euclidean([self.Data[0][i],self.Data[1][i]],[self.Data[0][j],self.Data[1][j]]))
                #dist[j][i]=dist[i][j]=(distance.euclidean([self.Data[0][i][ for ],[self.Data[0][j],self.Data[1][j]]))
            if i==j:
                dist[i][j]=0
            else:
                continue
    return dist