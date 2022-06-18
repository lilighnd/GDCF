import numpy as np
import pandas as pd
from sklearn import cluster
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

class Plot:
    def __init__(self,data,labels):
        self.x=[data[0][i] for i in range(len(data[0]))]       
        self.y=[data[1][i] for i in range(len(data[0]))]
        self.labels=labels

    def plot(self):
        info={'x':self.x,'y':self.y,'label':self.labels}
        df=pd.DataFrame(info)
        sns.scatterplot(x='x',y='y',hue='label',palette='Accent',data=df)
        plt.show()


    def plotClusters(self):
        info={'x':self.x,'y':self.y,'label':self.labels}
        df=pd.DataFrame(info)
        sns.scatterplot(x='x',y='y',hue='label',palette='Accent',data=df)
        plt.show()

    def plotPredLabel(self):
        print('m')