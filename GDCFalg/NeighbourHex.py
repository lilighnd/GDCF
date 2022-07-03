import numpy as np
import math
from GDCF import *


class NeighbourHex():

    def __init__(self, G, dim, b, numGx):
        self.g = G
        self.d = dim
        self.B = b  # HGB
        self.numGx = numGx
    def NeighbourGrid(self, Grids):
        n = len(self.B[0][0])
        Q = []
        tmp1 = np.ones((1, n))
        print(f"tmp1 : {len(self.g)}")
        print(f"tmp1 : {len(tmp1[0])}")
        for i in range(self.d):
            tmp2 = np.zeros((1, n))
            L1 = self.g[i]-int(np.ceil(math.sqrt(self.d)))
            L2 = self.g[i]+int(np.ceil(math.sqrt(self.d)))
            if L1 < 1:
                L1 = 0
            if L2 >= len(self.B[i]):
                L2 = len(self.B[i])-1
            for j in range(L1, L2+1):
                tmp2 = ORarray(self.B[i][j][:], tmp2)

            tmp1 = ANDarray(tmp1, tmp2)
        i = 0

        if self.g[0] - 2 >= 0: 
            x = self.g[0] - 2
            y = self.g[1] - 2
            # g = x * self.numGx + y
            # if g <= len(tmp1):
            print(f"Grids : {Grids,type(Grids),len(Grids)}")
            # tmp1[g] = 0
            # print(f"NHex x,y : {x,y,self.numGx,g,tmp1 }")

            x = self.g[0] - 2
            y = self.g[1] + 2
            g = x * self.numGx + y
            if g <= len(tmp1):
                tmp1[g] = 0

            x = self.g[0] + 2
            y = self.g[1] - 2
            g = x * self.numGx + y
            if g <= len(tmp1):
                tmp1[g] = 0

            x = self.g[0] +2 
            y = self.g[1] -1
            g = x * self.numGx + y
            if g <= len(tmp1):
                tmp1[g] = 0

            x = self.g[0] + 2
            y = self.g[1] + 1
            g = x * self.numGx + y
            if g <= len(tmp1):
                tmp1[g] = 0

            x = self.g[0] + 2
            y = self.g[1] + 2
            g = x * self.numGx + y
            if g <= len(tmp1):
                tmp1[g] = 0
        # m=[Grids[i][0] for i in range(len(Grids))]#Number  of grids
        for j in range(0, n):
            # and 'g'+str(j+1)!=self.g[0]:#g is neighbour of self
            if tmp1[0][j] == 1 and Grids[j] != []:
                # Q.append('g'+str(j+1))
                Q.append(Grids[j])

        return Q


def ORarray(a, b):
    for k in range(len(b[0])):
        if b[0][k] == 1 or a[k] == 1:
            b[0][k] = 1
        elif b[0][k] == 0 and a[k] == 0:
            b[0][k] = 0
    return b


def ANDarray(a, b):
    for k in range(len(b[0])):
        if b[0][k] == 1 and a[0][k] == 1:
            b[0][k] = 1
        elif b[0][k] == 0 or a[0][k] == 0:
            b[0][k] = 0
    return b


def GetGridByID(id, Grids):
    if id:
        return Grids[id]
    else:
        return 0
