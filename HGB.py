import numpy as np
class HGB():
    """HGB PROCCESS"""
    def __init__(self,s,dim):
        self.S=s
        self.d=dim
    def BuildHGB(self):
        k=0
        B=[]
        Max=[0,0]
        for i in range(self.d):
            for j in range(len(self.S)):
                if self.S[j][0]>Max[0]:
                    Max[0]=self.S[j][0]
                if self.S[j][1]>Max[1]:
                    Max[1]=self.S[j][1]
            #b=np.zeros(((self.S[-1][i]+1),len(self.S)))
            b=np.zeros((Max[i]+1,len(self.S)))
            B.append(b)

        for g in self.S:
            if len(g)==3: 
                for i in range(self.d):
                    pos=g[i]
                    B[i][pos][k]=1
            k+=1
        return B
                    



