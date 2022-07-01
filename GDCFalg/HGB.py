import numpy as np
class HGB():
    """HGB PROCCESS"""
    def __init__(self,s,dim):
        self.Grids=s
        self.d=dim
    def BuildHGB(self):
        k=0
        B=[]
        Max=[0,0]
        for i in range(self.d):
            for j in range(len(self.Grids)):
                if self.Grids[j][0]>Max[0]:
                    Max[0]=self.Grids[j][0]
                if self.Grids[j][1]>Max[1]:
                    Max[1]=self.Grids[j][1]
            #b=np.zeros(((self.S[-1][i]+1),len(self.S)))
            b=np.zeros((Max[i]+1,len(self.Grids)))
            B.append(b)

        for g in self.Grids:
            if len(g)==3: 
                for i in range(self.d):
                    pos=g[i]
                    B[i][pos][k]=1
            k+=1
        print(f"hgb : {B[0][4][0]}")
        print(f"hgb : {B[0][4][1]}")
        print(f"hgb : {B[0][4][2]}")
        print(f"hgb : {B[0][4][3]}")
        print(f"hgb : {B[0][4][4]}")
        print(f"hgb : {B[0][4][5]}")
        print(f"hgb : {B[0][4][6]}")
        print(f"hgb : {B[0][4][7]}")
        print(f"hgb : {B[0][4][8]}")
        print(f"hgb : {B[0][4][9]}")
        print(f"hgb : {B[0][4][10]}")
        print(f"hgb : {B[0][4][11]}")
        print(f"hgb : {B[0][4][12]}")
        return B
                    



