import pandas as pd

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
    def Heartdata(cls):
        path = f'..\DataSet\HeartData.csv'
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
    def PizzaData(cls):
        path = f'..\DataSet\Pizza.csv'
        df = pd.read_csv(path)
        data = df.values.tolist()
        for i in range(len(data)):
            data[i] = data[i][1:8]
        return cls(data)

    @classmethod
    def Test30(cls):
        True_label=[]
       #path = f'./blobs/blobsData.csv'
        path = f'.\\blobsData.csv'
        df = pd.read_csv(path)
        data = df.values.tolist()
        for i in range(len(data)):
            True_label.append(data[i][-1])
            data[i] = data[i][0:2]

        return cls(data),True_label

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