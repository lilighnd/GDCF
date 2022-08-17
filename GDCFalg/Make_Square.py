import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib.patches as patches
from Plot import *
import time
##==============================
class Make_Square():
    def __init__(self,D,e,dim):
        self.dim=dim
        self.d=D
        self.Data=np.transpose(np.array(D))
        self.Eps=e
    def GridHex(self):
        st_time_partSq=time.time()

        LengthCell = (self.Eps/np.sqrt(2))


        maxX = max(self.Data[:,0])  # ==> 26
        minX = min(self.Data[:,0]) # ==> 1min(Data[0])

        maxY = max(self.Data[:,1])  # ==> 21
        minY = min(self.Data[:,1])  # ==> 1min(Data[1])

        DataInGrid=[]
        flg=True
        numG=[]
        point = np.array([minX,minY])
        #numGridX = int(np.ceil((maxX-minX)/(3*self.Eps)))*2  # ==> 5
        numGridX =int(np.ceil(((maxX-minX)/LengthCell)))  # ==> 5
        #numGridY = int(np.ceil((maxY-minY)/(2*self.Eps)))  # ==> 4
        numGridY = int(np.ceil((maxY-minY)/LengthCell)) # ==> 4
        GX=int(numGridX * numGridY)
        #self.Eps=2*self.Eps
        #for j in range(numGridY):
        #    for i in range(numGridX):
        #        numG.append([((i+1)+(j*numGridX)),i,j])
        d1=[]
        d2=[]
        for i in range(numGridY):
            if i != 0:
                point=temp[2]
                flg=True

            for j in range(numGridX):
                hex1 = self.ahex(point)
                n1 = self.hexn(point)
                
                #------for plot----------
                for k in range(len(hex1)):
                    d1.append(hex1[k][0])
                    d2.append(hex1[k][1])
                #------------------------
                if j==0:
                    temp=n1
                #n1hx = []
                #for i in range(len(n1)):
                #    n1hx.append(self.ahex(n1[i]))
                #n1hx = np.array(n1hx)
                poly = path.Path(hex1)
                q=0
                inner_d = poly.contains_points(self.Data)
                inner_data = np.where(inner_d)
                # print(f"inner data[0] : {inner_data[0]}")
                DataInGrid.append(inner_data[0])
                # print(DataInGrid)

                # if flg:
                point=n1[0]
                # else:
                #     point=n1[2]
                # flg=not flg
                numG.append([j,i])
                #if inner_data!=[] and len(numG[-1])==2:#Determine Not Empty Grids
                if len(inner_data)!=0 and len(numG[-1])==2:#Determine Not Empty Grids
                    numG[-1].append("Not Empty Grid")
                #print(inner_data)


        #fig = plt.figure()
        #ax = fig.add_subplot(1, 1, 1)
        #data=[d1,d2]

        #plot0=Plot(self.d,1)
        #p0=plot0.plot()

        #plot1=Plot(data,0)
        #p1=plot1.plot()

        
        #plt.show()
        data=[d1,d2]
        
        """fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = fig.add_subplot(111)
        ax1.scatter(d1, d2, s=1, c='b', marker="s", label='first')
        ax2.scatter(self.d[0],self.d[1], s=10, c='r', marker="o", label='second')
        ax1.set_xlim([0,30])
        ax2.set_ylim([0,30])
        plt.legend(loc='upper left');
        plt.show()"""
       
        

        NonEmptyGrid=[]
        dim_Grids=[]
        for i in range(len(DataInGrid)):
            if DataInGrid[i]!=[]:
                dim_Grids.append(numG[i])
                NonEmptyGrid.append(DataInGrid[i])

        time_partSq=time.time()-st_time_partSq
        print(f"dim_grids : {dim_Grids}")   
        print(f"NonEmptyGrid : {NonEmptyGrid}")   
        return dim_Grids,NonEmptyGrid
        #*****************************plt.show()


    
    def ahex(self,p):
        # h = [[round(p[0] + self.Eps/2 *np.cos(np.pi/4),2)
        #      ,round(p[1] + self.Eps/2 *np.cos(np.pi/4),2)]]
        # for i in range(3):
        #     x = h[i][0] + (self.Eps/np.sqrt(2))*(np.cos((i*(np.pi/2)) + np.pi))
        #     y = h[i][1] + (self.Eps/np.sqrt(2))*(np.sin((i*(np.pi/2)) + np.pi))


        # h=[]
        # for i in range(1,7):
        #     angle_deg = 90 * i - 45
        #     angle_rad = np.pi / 180 * angle_deg
        #     x = p[0] + (self.Eps/2) * np.cos(angle_rad)
        #     y = p[1] + (self.Eps/2) * np.sin(angle_rad)
        #     # h.append(np.round([x, y], 2))
        #     h.append([x, y])
        # #     h.append(np.round([x,y],2))



        h=[]
        x = p[0] + self.Eps/(2*np.sqrt(2))
        y = p[1] + self.Eps/(2*np.sqrt(2))
        h.append([x, y])

        x = p[0] - self.Eps/(2*np.sqrt(2))
        y = p[1] + self.Eps/(2*np.sqrt(2))
        h.append([x, y])
        
        x = p[0] - self.Eps/(2*np.sqrt(2))
        y = p[1] - self.Eps/(2*np.sqrt(2))
        h.append([x, y])
        
        x = p[0] + self.Eps/(2*np.sqrt(2))
        y = p[1] - self.Eps/(2*np.sqrt(2))
        h.append([x, y])
        return np.array(h)
    ##------------------------------
    def hexn(self,p):
        h=[]
        x = p[0] + self.Eps/np.sqrt(2)
        y = p[1] 
        h.append([x, y])

        x = p[0] + self.Eps/np.sqrt(2)
        y = p[1] + self.Eps/np.sqrt(2)
        h.append([x, y])
        
        x = p[0] 
        y = p[1] + self.Eps/np.sqrt(2)
        h.append([x, y])
        
        x = p[0] - self.Eps/np.sqrt(2)
        y = p[1] + self.Eps/np.sqrt(2)
        h.append([x, y])
        
        x = p[0] - self.Eps/np.sqrt(2)
        y = p[1]
        h.append([x, y])
        
        x = p[0] - self.Eps/np.sqrt(2)
        y = p[1] - self.Eps/np.sqrt(2)
        h.append([x, y])
        
        x = p[0] 
        y = p[1] - self.Eps/np.sqrt(2)
        h.append([x, y])
        
        x = p[0] + self.Eps/np.sqrt(2)
        y = p[1] - self.Eps/np.sqrt(2)
        h.append([x, y])
        
       
       
        # h=[[] for x in range(8)]
        # for i in range(1,5):
        #     angle_deg = 90 * i - 90
        #     angle_rad = np.pi / 180 * angle_deg
        #     x = p[0] + (np.sqrt(2)/2*self.Eps) * np.cos(angle_rad)
        #     y = p[1] + (np.sqrt(2)/2*self.Eps) * np.sin(angle_rad)
        #     # h.append(np.round([x, y], 2))
        #     h[2*i-2].append([x, y]) 

        # for i in range(1,5):
        #     angle_deg = 90 * i - 45
        #     angle_rad = np.pi / 180 * angle_deg
        #     x = p[0] + self.Eps * np.cos(angle_rad)
        #     y = p[1] + self.Eps * np.sin(angle_rad)
        #     # h.append(np.round([x, y], 2))
        #     h[2*i-1].append([x, y])    










        #h = [[p[0]+(self.Eps/(np.sqrt(2))*np.cos(0)),p[1]+(self.Eps/(np.sqrt(2))*np.sin(0))]]
        #x = h[0][0]+(self.Eps/(np.sqrt(2))*np.cos(0))
        #y = h[0][1]+(self.Eps/(np.sqrt(2))*np.sin(np.pi/2))
        #h.append(np.round([x,y],2))
        
        

        #for i in range(6):
        #    x = h[i][0] + (self.Eps/(np.sqrt(2))*np.cos(np.pi/2+(np.pi/2*i)))
        #    y = h[i][1] + (self.Eps/(np.sqrt(2))*np.sin(np.pi/2+(np.pi/2*i)))
        #    h.append(np.round([x,y],2))
        #    x = h[i][0] + (self.Eps/(np.sqrt(2))*np.cos(np.pi/2+(np.pi/2*i)))
        #    y = h[i][1] + (self.Eps/(np.sqrt(2))*np.sin(np.pi/2+(np.pi/2*i)))
        #    h.append(np.round([x,y],2))



        # #First Neighbour Center
        # h = [[round((p[0] + self.Eps/(np.sqrt(2))*np.cos(0)),2)
        #      ,round((p[1] + self.Eps/(np.sqrt(2))*np.sin(0)),2)]]

        # #Second Neighbour Center
        # x = h[0][0] + (self.Eps/(np.sqrt(2))*np.cos(np.pi/2))
        # y = h[0][1] + (self.Eps/(np.sqrt(2))*np.sin(np.pi/2))
        # h.append(np.round([x,y],2))
        # k=1
        # for i in range(3):
        #     #for j in range(2):
        #     x = h[i+k][0] + (self.Eps/(np.sqrt(2))*np.cos(np.pi+((np.pi/2)*i)))
        #     y = h[i+k][1] + (self.Eps/(np.sqrt(2))*np.sin(np.pi+((np.pi/2)*i)))
        #     h.append(np.round([x,y],2))
        #     k+=1
        #     x = h[i+k][0] + (self.Eps/(np.sqrt(2))*np.cos(np.pi+((np.pi/2)*i)))
        #     y = h[i+k][1] + (self.Eps/(np.sqrt(2))*np.sin(np.pi+((np.pi/2)*i)))
        #     h.append(np.round([x,y],2))
        
        return(np.array(h))
##------------------------------

##==============================
#d1 = np.random.normal(0, 10, (500,2))
#d2 = np.random.normal(20, 5, (250,2))
#data = np.concatenate((d1,d2))


#point = np.array([0,0])
#self.Eps = 3
#hex1 = ahex(point,self.Eps)
#n1 = hexn(point,self.Eps)

#n1hx = []
#for i in range(len(n1)):
#    n1hx.append(ahex(n1[i],self.Eps))
#n1hx = np.array(n1hx)

###plt.plot(data[:,0],data[:,1],".")
###for i in range(6):
###    plt.plot(n1hx[i,:,0],n1hx[i,:,1],"m")
###plt.plot(hex1[:,0],hex1[:,1],"r")
###plt.show()

#poly = path.Path(hex1)

#inner_data = poly.contains_points(data)
#inner_data, = np.where(inner_data)
#print(inner_data)

#fig, ax = plt.subplots()
#patch = patches.PathPatch(poly, edgecolor="m",facecolor="none")
#ax.add_patch(patch)
#ax.set_xlim(min(data[:,0])-10,max(data[:,0])+10)
#ax.set_ylim(min(data[:,1])-10,max(data[:,1])+10)
#ax.plot(data[:,0],data[:,1],".")
#plt.show()




#---------------------------------------------square n-dim---------------------------------------------------
    def GridSqn(self):
        LengthCell = (self.Eps/np.sqrt(self.dim))
        numberOfGrid=[]
        dim_Grids1=[]
        NonEmptyGrid1=[]
        x=[]
        for d in range(self.dim):
            Max = max(self.Data[:,d])  
            Min = min(self.Data[:,d])   
            num_G=int(np.ceil(((Max+0.01)-(Min-0.01))/LengthCell))+1
            if num_G<0:
                numberOfGrid.append(0)
            numberOfGrid.append(num_G)
            print(f"min,max,numberOfGrid:{Min,Max,num_G}")
        print(f"numg : {numberOfGrid,x}")


    
        for i in range(len(self.Data)):
            DimGrid=[]
            for j in range(self.dim):
                Number_Grid=int((np.ceil(self.Data[i][j]/LengthCell)))-1
                print(f"Number_Grid : {i,j,Number_Grid}")
                DimGrid.append(Number_Grid)
            if DimGrid not in dim_Grids1:
                dim_Grids1.append(DimGrid)
                NonEmptyGrid1.append([i])
            else:
                for k in range(len(dim_Grids1)):
                    if DimGrid==dim_Grids1[k]:
                        ind=k
                        break
                # ind=dim_Grids.index(DimGrid)
                NonEmptyGrid1[ind].append(i)

        NonEmptyGrid=[]
        dim_Grids=[]
        for i in range(len(NonEmptyGrid1)):
            if NonEmptyGrid1[i]!=[]:
                dim_Grids.append(dim_Grids1[i])
                NonEmptyGrid.append(NonEmptyGrid1[i])
        print(f"dim_grids : {dim_Grids}")   
        print(f"NonEmptyGrid : {len(NonEmptyGrid)}")  
        return dim_Grids,NonEmptyGrid,numberOfGrid

        


























