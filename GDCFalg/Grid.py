import math
import numpy as np
class Grid:
    def __init__(self,data,eps,d):
        self.D=data
        self.x=[data[0][i] for i in range(len(data[0]))]       
        self.y=[data[1][i] for i in range(len(data[1]))]
        self.Eps=eps
        self.dim=d
        
    def GridDim(self):
        LengthCell = (self.Eps/(math.sqrt(self.dim)))
        maxX = max(self.D[0]) 
        minX =0  
        maxY = max(self.D[1]) 
        minY =0  
        numGridX = int(np.ceil((maxX-minX)/LengthCell))  
        numGridY = int(np.ceil((maxY-minY)/LengthCell))  
        GX=int(numGridX * numGridY)
        G=[[[] for i in range(numGridY)] for j in range(numGridX)]
        numG=[]#number and coordinates all of Grids
        nonEmptyGrids=[]
        x=0
        for j in range(numGridY):
            for i in range(numGridX):
                #numG.append([((i+1)+(j*numGridX)),i,j])New
                numG.append([i,j])

        DataInGrids=[[] for i in range(len(numG))]
        for i in range(len(self.D[0])):
            g=int((np.ceil(self.D[0][i]/LengthCell))+((numGridX)*(np.ceil(self.D[1][i]/LengthCell)-1)))
            #DataInGrids[g].append([self.D[0][i],self.D[1][i]])New
            DataInGrids[g].append(i)
            if DataInGrids[g]!=[] and len(numG[g])==2:
                numG[g].append("Not Empty Grid")
        

        NonEmptyGrid=[]
        dim_Grids=[]
        for i in range(len(DataInGrids)):
            if DataInGrids[i]!=[]:
                dim_Grids.append(numG[i])
                NonEmptyGrid.append(DataInGrids[i])
        

        return dim_Grids,NonEmptyGrid


