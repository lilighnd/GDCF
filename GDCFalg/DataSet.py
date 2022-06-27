from tkinter import Y
import pandas as pd
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn import cluster, datasets

class DataSet:
    def __init__(self, data):
        self.Data = data
    
    def __str__(self):
        return f"[DataSet: {len(self.Data)} item(s)]"


    @classmethod
    def data3(cls):
        path = f'..\DataSet\data3.csv'
        df = pd.read_csv(path)
        tmpdata = df.values.tolist()
        for i in range(len(tmpdata)):
            tmpdata[i] = tmpdata[i][0:2]
        return tmpdata




    @classmethod
    def data(cls):
        path = f'..\DataSet\data1.csv'
        df = pd.read_csv(path)
        tmpdata = df.values.tolist()
        for i in range(len(tmpdata)):
            tmpdata[i] = tmpdata[i][0:2]
        return tmpdata

    @classmethod
    def data2(cls):
        path = f'..\DataSet\data2.csv'
        df = pd.read_csv(path)
        tmpdata = df.values.tolist()
        for i in range(len(tmpdata)):
            tmpdata[i] = tmpdata[i][0:2]
        return tmpdata

    @classmethod
    def bigdata(cls):
        path = f'..\DataSet\pts900class4.csv'
        df = pd.read_csv(path)
        tmpdata = df.values.tolist()
        for i in range(len(tmpdata)):
            tmpdata[i] = tmpdata[i][0:2]
        return tmpdata


    @classmethod
    def Crescentdata(cls):
        path = f'..\DataSet\CrescentData.csv'
        df = pd.read_csv(path)
        tmpdata = df.values.tolist()
        for i in range(len(tmpdata)):
            tmpdata[i] = tmpdata[i][0:2]
        return tmpdata

   

    @classmethod
    def IrisOnlyInput(cls):
        path = f'..\DataSet\iris.csv'
        df = pd.read_csv(path)
        data = df.values.tolist()
        for i in range(len(data)):
            data[i] = data[i][0:4]
        return data  
    
    @classmethod
    def Iris(cls):
        path = f'..\DataSets\iris.csv'
        df = pd.read_csv(path)
        data = df.values.tolist()
        for i in range(len(data)):
            data[i] = data[i][4]
        return cls(data)

    

    @classmethod
    def Test30(cls):
        True_label=[]
        #path = f'/content/blobs/blobsData1m.csv'
        # path = f'/content/drive/MyDrive/Colab Notebooks/blobsData1m.csv'
        path = f'/content/drive/MyDrive/Colab Notebooks/moonsData38.csv'
        df = pd.read_csv(path)
        data = df.values.tolist()

        
        # my_path = "blobsData1m.csv"
        # cd=os.getcwd()
        # df = pd.read_csv(cd +"\\GDCFalg\\" + my_path)
        # data = df.values.tolist()
        # for i in range(len(data)):
        for i in range(len(data)):
            True_label.append(data[i][-1])
            data[i] = data[i][0:2]
            
        
        

        # path = f'/content/drive/MyDrive/Colab Notebooks/blobsLabel1m.csv'
        path = f'/content/drive/MyDrive/Colab Notebooks/moonsLabels38.csv'
        df = pd.read_csv(path)
        True_label = df.values.tolist()
        # for i in range(len(True_label)):
        for i in range(len(True_label)):
            True_label[i] = True_label[i][0]


        X_train,X_test,Y_train,Y_test = train_test_split(data,True_label,test_size=0.90,random_state=42)
        print(X_test,Y_test)

        return cls(X_test),Y_test
        # return cls(data),True_label

    @classmethod
    def moons1m(cls):
        True_label=[]
        moons = datasets.make_moons(n_samples=1000000,noise=0.05,random_state=42)
        True_label = moons[1]
        Data=moons[0] 
        # X_train,X_test,Y_train,Y_test = train_test_split(moons,True_label,test_size=1,random_state=42)
        print(f"Data and labels :{Data,True_label}")
        return cls(Data),True_label   
        # return cls(X_test),Y_test   
    @classmethod
    def d1(cls):
        True_label=[]
        path = f'..\\DataSet\\cassini.csv'#0.07,3
        df = pd.read_csv(path)
        data = df.values.tolist()
        for i in range(len(data)):
            True_label.append(data[i][-1])
            data[i] = data[i][0:2]
        return cls(data),True_label