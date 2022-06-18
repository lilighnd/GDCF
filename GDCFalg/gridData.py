import math
import numpy as np
class Partitioning:
    def __init__(self,data,eps,d):
        self.D=data
        self.d1=[data[i][0] for i in range(len(data))]       
        self.d2=[data[i][1] for i in range(len(data))]
        self.d=[d1,d2]
        self.Eps=eps
        self.dim=d

    def partitions(self):
        self.minmax=np.zeros(d,2)
        for j in range(0,1):
            for i in range(1,d):
                self.minmax[i,0]=min(self.d[j])+0.01
                self.minmax[i,1]=max(self.d[j])-0.01

        self.lengthcell=self.Eps/math.sqrt(self.dim)
        
        dim1=self.grids(1)
        dim2=self.grids(2)
        part1=[[i+1] for i in range(len(dim1))]
        part2=[[i+1] for i in range(len(dim2))]
        parts=part1*part2
        for i in range(len(self.D)):
            for j in range(self.dim):

                if 


    def grids(self,dim):
        grid=[]
        tmp=self.minmax[dim][0]
        while tmp<= self.minmax[dim][1]:   
            grid.append(tmp)
            tmp+=self.lengthcell
