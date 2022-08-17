import numpy as np
class HGB():
    """HGB PROCCESS"""
    def __init__(self,s,dim,numGrid_dim):
        self.rowHGB=numGrid_dim
        self.nonemptyGrids=s
        self.d=dim
    def BuildHGB(self):
        print(f"nonempty : {len(self.nonemptyGrids)}")
        k=0
        B=[]
        Max=[0,0]
        # for i in range(self.d):
        #     for j in range(len(self.nonemptyGrids)):
        #         if self.nonemptyGrids[j][0]>Max[0]:
        #             Max[0]=self.nonemptyGrids[j][0]
        #         if self.nonemptyGrids[j][1]>Max[1]:
        #             Max[1]=self.nonemptyGrids[j][1]
        #     b=np.zeros((Max[i]+1,len(self.nonemptyGrids)))
        #     B.append(b)
        print(f"self.rowHGB:{self.rowHGB}")
        for i in range(self.d):
            b=np.zeros((self.rowHGB[i],len(self.nonemptyGrids)))
            B.append(b)
        

        # numG[numGX,numGY]
        # for i in range(self.d):
        #     b=np.zeros((numG[i]+1,len(self.nonemptyGrids)))
        #     B.append(b)

        for g in self.nonemptyGrids:
            for i in range(self.d):
                print(f"g,i : {g,i}")
                pos=g[i]
                B[i][pos][k]=1
                print(f"i,pos,k : {i,pos,k}")

            k+=1
        print(f"B : {B[0][3][0]}")
        print(f"B : {B[0][3][1]}")
        print(f"B : {B[0][3][2]}")
        print(f"B : {B[0][3][3]}")
        print(f"B : {B[0][3]}")
        return B
                    



