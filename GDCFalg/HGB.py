import numpy as np
class HGB():
    """HGB PROCCESS"""
    def __init__(self,s,dim):
        self.nonemptyGrids=s
        self.d=dim
    def BuildHGB(self):
        k=0
        B=[]
        Max=[0,0]
        for i in range(self.d):
            for j in range(len(self.nonemptyGrids)):
                if self.nonemptyGrids[j][0]>Max[0]:
                    Max[0]=self.nonemptyGrids[j][0]
                if self.nonemptyGrids[j][1]>Max[1]:
                    Max[1]=self.nonemptyGrids[j][1]
            b=np.zeros((Max[i]+1,len(self.nonemptyGrids)))
            B.append(b)

        # numG[numGX,numGY]
        # for i in range(self.d):
        #     b=np.zeros((numG[i]+1,len(self.nonemptyGrids)))
        #     B.append(b)


        for g in self.nonemptyGrids:
            for i in range(self.d):
                pos=g[i]
                B[i][pos][k]=1
            k+=1
        # print(f"hgb : {B}")
        return B
                    



