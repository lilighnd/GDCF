import numpy as np
import math
from GDCF import *


class NeighbourHex():

    def __init__(self, G, dim, b):
        self.g = G
        self.d = dim
        self.B = b  # HGB
    def NeighbourGrid(self, Grids):
        print("********************start NHEX******************")
        # print(Grids)
    
        
        n = len(self.B[0][0])
        Q = []
        tmp1 = np.ones((1, n))
        # print(f"tmp1 : {len(tmp1[0])}")
        # print(f"tmp1 : {tmp1[0]}")
        for i in range(self.d):
            tmp2 = np.zeros((1, n))
            L1 = self.g[i]-int(np.ceil(math.sqrt(self.d)))
            L2 = self.g[i]+int(np.ceil(math.sqrt(self.d)))
            if L1 < 1:
                L1 = 0
            if L2 >= len(self.B[i]):
                L2 = len(self.B[i])-1
            for j in range(L1, L2+1):
                # tmp2 = ORarray(self.B[i][j][:], tmp2)
                tmp2 = np.logical_or(self.B[i][j][:], tmp2)
            # tmp1 = ANDarray(tmp1, tmp2)
            tmp1 = np.logical_and(tmp1, tmp2)

        i = 0


        x = self.g[0] - 2
        y = self.g[1] - 2
        if [x, y, 'Not Empty Grid'] in Grids:
            indx = Grids.index([x, y, 'Not Empty Grid'])
            # print(f"tmp1 start: {tmp1[0][indx]}")
            # print(f"tmp1 start: {tmp2[0]}")
            tmp1[0][indx] = 0 
            # print(f"indx and grid: {indx,[x, y, 'Not Empty Grid']}")
            # print(f"tmp1 end: {tmp1[0][indx]}")
        # print(f"grids : {Grids}")
        x = self.g[0] - 2
        y = self.g[1] + 2
        if [x, y, 'Not Empty Grid'] in Grids:
            indx = Grids.index([x, y, 'Not Empty Grid'])
            tmp1[0][indx] = 0 

        x = self.g[0] + 2
        y = self.g[1] - 2
        if [x, y, 'Not Empty Grid'] in Grids:
            indx = Grids.index([x, y, 'Not Empty Grid'])
            tmp1[0][indx] = 0 

        x = self.g[0] +2 
        y = self.g[1] -1
        if [x, y, 'Not Empty Grid'] in Grids:
            indx = Grids.index([x, y, 'Not Empty Grid'])
            tmp1[0][indx] = 0

        x = self.g[0] + 2
        y = self.g[1] + 1
        if [x, y, 'Not Empty Grid'] in Grids:
            indx = Grids.index([x, y, 'Not Empty Grid'])
            tmp1[0][indx] = 0 

        x = self.g[0] + 2
        y = self.g[1] + 2
        if [x, y, 'Not Empty Grid'] in Grids:
            indx = Grids.index([x, y, 'Not Empty Grid'])
            tmp1[0][indx] = 0 
        # m=[Grids[i][0] for i in range(len(Grids))]#Number  of grids
        for j in range(0, n):
            # and 'g'+str(j+1)!=self.g[0]:#g is neighbour of self
            if tmp1[0][j] == 1 and Grids[j] != []:
                # Q.append('g'+str(j+1))
                Q.append(Grids[j])

        return Q



def GetGridByID(id, Grids):
    if id:
        return Grids[id]
    else:
        return 0
