import numpy as np
import math
from GDCF import *
import time
class NeighbourGridQuery():

    def __init__(self,G,dim,b):
        self.g=G
        self.d=dim
        self.B=b#HGB
    def NeighbourGrid(self,Grids):
        st_neisq=time.time()
        # print("-------------square grid neighbour------------")
        st_neiSq=time.time()
        n=len(self.B[0][0])
        # print(f"n : {n}")
        Q=[]
        tmp1=np.ones((1,n))
        for i in range(self.d):
            tmp2=np.zeros((1,n))
            L1=self.g[i]-int(np.ceil(math.sqrt(self.d)))
            L2=self.g[i]+int(np.ceil(math.sqrt(self.d)))
            if L1<1:
                L1=0
            if L2>=len(self.B[i]):
                L2=len(self.B[i])-1
            for j in range(L1,L2+1):
                tmp2=np.logical_or(self.B[i][j][:],tmp2)
                # print(f"L1,L2,tmp2:{L1,L2,tmp2}")
            tmp1=np.logical_and(tmp1,tmp2)
            # print(f"tmp1:{tmp1}")
        i=0
        

        #m=[Grids[i][0] for i in range(len(Grids))]#Number  of grids
        for j in range(0,n):
            if tmp1[0][j]==1 and Grids[j]!= []: #and 'g'+str(j+1)!=self.g[0]:#g is neighbour of self
                #Q.append('g'+str(j+1))
                Q.append( Grids[j] )
        # time_neiSq=time.time()-st_neiSq 
        # time_neisq=time.time()-st_neisq
        # print(f"Q:{Q}")
        return Q


def ORarray(a,b):
    for k in range(len(b[0])):
        if b[0][k]==1 or a[k]==1:
            b[0][k]=1
        elif b[0][k]==0 and a[k]==0:
            b[0][k]=0
    return b

def ANDarray(a,b):
    for k in range(len(b[0])):
        if b[0][k]==1 and a[0][k]==1:
            b[0][k]=1
        elif b[0][k]==0 or a[0][k]==0:
            b[0][k]=0
    return b

def GetGridByID(id,Grids):
    if id :
        return Grids[id]
    else:
        return 0
        
