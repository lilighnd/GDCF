from tkinter import Y
import pandas as pd
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn import cluster, datasets
import json
# import xlsxwriter
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
import time
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
    def datad1(cls):
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
    def dataclutot(cls):
        True_label=[]
        path = f'/content/drive/MyDrive/aggregation.csv'
        df = pd.read_csv(path)
        data = df.values.tolist()

        
        for i in range(len(data)):
            True_label.append(data[i][-1])
            data[i] = data[i][0:2]
            
    
        return cls(data),True_label

    @classmethod
    def data(cls):
        with open('/content/drive/MyDrive/Colab Notebooks/inputobject.json', 'r') as openfile:
            # Reading from json file
            json_object = json.load(openfile)

        D = json_object["data"]
        Numbers = int(json_object["n_samples"])
        Noise = float(json_object["noise"])
        R = int(json_object["random_state"])
        f = int(json_object["features"])
        c = int(json_object["centers"])


        True_label=[]
        if D == "moon":
            print("moons data")       
            data = datasets.make_moons(n_samples=Numbers,noise=Noise,random_state=R)
            excel_name_label = f'/content/drive/MyDrive/Colab Notebooks/mydata.xlsx'
            df_data = pd.DataFrame(data[0])
            df_data.to_excel(excel_name_label,index=False)



            # excel_name = f'/content/drive/MyDrive/Colab Notebooks/mydata.xlsx'

            # df1=pd.DataFrame(data[0],columns=['Name', 'M-cap'])
            # df2=pd.DataFrame(data[1],columns=['l'])
            # dfs = [df1,df2]
            # df = pd.concat(dfs,axis=1, join='inner')
            # df.to_excel(excel_name,index=False)




        if D == "blob":
            print("blobs data")       
            data = datasets.make_blobs(n_samples=Numbers, n_features = f, 
                  centers = 3,cluster_std = 1,random_state=R)
            excel_name_label = f'/content/drive/MyDrive/Colab Notebooks/mydata.xlsx'
            df_data = pd.DataFrame(data[0])
            df_data.to_excel(excel_name_label,index=False)


        if D == "blobnd":
            print("blobs data")       
            data = datasets.make_blobs(n_samples=Numbers, n_features = f, 
                    centers = 3,cluster_std = 1,random_state=R)

            t1=time.time()
            x = LocallyLinearEmbedding(n_components=2)
            X_transformed = x.fit_transform(data[0][:Numbers])
            t2=time.time()-t1
            print(f"time locally linear:{t2}")
            excel_name_label = f'/content/drive/MyDrive/Colab Notebooks/mydatablob3d.xlsx'
            datand=X_transformed
            df_data = pd.DataFrame(data[0])
            df_data.to_excel(excel_name_label,index=False)


        if D == "circle":
            print("circles data")       
            data = datasets.make_circles(n_samples=Numbers,noise=Noise,random_state=R,factor=0.5)
            excel_name_label = f'/content/drive/MyDrive/Colab Notebooks/mydata.xlsx'
            df_data = pd.DataFrame(data[0])
            df_data.to_excel(excel_name_label,index=False)

        True_label = data[1]
        Data=data[0] #for 1-dim
        # Data=datand#for n-dimention data
        # X_train,X_test,Y_train,Y_test = train_test_split(moons,True_label,test_size=1,random_state=42)
        # print(f"Data and labels :{Data,len(True_label)}")
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