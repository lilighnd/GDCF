import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib.patches as patches
from Plot import *
import math
# ==============================


class make_Hex():
    def __init__(self, D, e):
        self.d = D
        self.Data = np.transpose(np.array(D))
        self.Eps = e

    def GridHex(self):
        LengthCell = (self.Eps*2)

        maxX = max(self.Data[:, 0])  # ==> 26
        minX = min(self.Data[:, 0])  # ==> 1min(Data[0])

        maxY = max(self.Data[:, 1])  # ==> 21
        minY = min(self.Data[:, 1])  # ==> 1min(Data[1])

        DataInGrid = []
        flg = True
        numG = []
        point = np.array([minX, minY])
        # print(f"point : {point}")
        # print(f"data : {self.Data}")

        # print(f"point : {point}")#-------
        # print(f"data : {np.array(self.Data)}")#-------
        numGridX = int(np.ceil(((maxX-minX)/(3*(self.Eps/2)))*2))  # ==> 5
        numGridY = int(np.ceil((maxY-minY)/(0.866*self.Eps)))  # ==> 4
        GX = int(numGridX * numGridY)
        # print(f"xgrid,ygrid,allgrid : {numGridX,numGridY,GX}")nnn
        # print(f"all grid : {GX}")
        d1 = []
        d2 = []
        for i in range(numGridY):
            # print(f"point : {point}")#-------
            # print(f"data : {np.array(self.Data)}")#-------
            # print(f"i and numgridy : {i,numGridY}")#-------
            if i != 0:
                point = temp[1]
                # print(f"point : {point}")#-------
                flg = True

            for j in range(numGridX):
                # print(f"j and numgridx : {j,numGridX}")#-------
                hex1 = self.ahex(point)
                n1 = self.hexn(point)
                # print(f"hex1 : {hex1}")  # -------nnn
                # print(f"n1 : {n1}")#-------nnn

                # ------for plot----------
                for k in range(len(hex1)):
                    d1.append(hex1[k][0])
                    d2.append(hex1[k][1])
                # ------------------------
                if j == 0:
                    temp = n1
                #n1hx = []
                # for i in range(len(n1)):
                #    n1hx.append(self.ahex(n1[i]))

                #n1hx = np.array(n1hx)
                poly = path.Path(hex1)
                inner_d = poly.contains_points(self.Data)
                inner_data = np.where(inner_d)
                DataInGrid.append(inner_data[0])
                # print(f"inner_d : {inner_d}")  # -------nnn
                # print(f"DataInGrid : {DataInGrid}")  # -------nnn

                if flg:
                    # print(f"flag true and point=n1[0] : {point}")#-------
                    point = n1[0]
                else:
                    # print(f"flag false and point=n1[-1] : {point}")#-------
                    point = n1[-1]
                flg = not flg
                numG.append([i, j])
                # print(f"lenght DataInGrid[-1] : {len(DataInGrid[-1])}")  # -------nnn
                # Determine Not Empty Grids
                # if len(DataInGrid[-1]) != 0 and len(numG[-1]) == 2:
                if len(DataInGrid[-1]) != 0:
                    numG[-1].append("Not Empty Grid")
                    # print(f"numG : {numG}")  # -------nnn

                # print(inner_data)

        #fig = plt.figure()
        #ax = fig.add_subplot(1, 1, 1)v
        # data=[d1,d2]

        # plot0=Plot(self.d,1)
        # p0=plot0.plot()

        # plot1=Plot(data,0)
        # p1=plot1.plot()

        # plt.show()
        # data=[d1,d2]
        #fig = plt.figure()
        #ax1 = fig.add_subplot(111)
        #ax2 = fig.add_subplot(111)
        #ax1.scatter(d1, d2, s=1, c='b', marker="s", label='first')
        #ax2.scatter(self.d[0],self.d[1], s=10, c='r', marker="o", label='second')
        # ax1.set_xlim([-1,1])
        # ax2.set_ylim([-1,1])
        #plt.legend(loc='upper left');
        # plt.show()

        NonEmptyGrid = []
        dim_Grids = []
        for i in range(len(DataInGrid)):
            if len(DataInGrid[i]) != 0:
                dim_Grids.append(numG[i])
                NonEmptyGrid.append(DataInGrid[i])
        # print(f"dim_Grids,NonEmptygrids : {dim_Grids,NonEmptyGrid}")nnn
        return dim_Grids, NonEmptyGrid
        # *****************************plt.show()

    def ahex(self, p):
        # h = [[p[0] + self.Eps/2, p[1]]]
        # for i in range(5):
        #     x = h[i][0] + (self.Eps/2*(np.cos((2+i)*np.pi/3)))
        #     y = h[i][1] + (self.Eps/2*(np.sin((2+i)*np.pi/3)))
        #     h.append(np.round([x, y], 2))
        # return np.array(h)

        h=[]
        for i in range(1,7):
            angle_deg = 60 * i - 60
            angle_rad = np.pi / 180 * angle_deg
            x = p[0] + (self.Eps/2) * np.cos(angle_rad)
            y = p[1] + (self.Eps/2) * np.sin(angle_rad)
            # h.append(np.round([x, y], 2))
            h.append([x, y])
        return np.array(h)
    # ------------------------------

    def hexn(self, p):
        # h = [[p[0]+1.5*self.Eps/2,
        #       np.round(p[1] + self.Eps/2*np.sin(np.pi/3), 2)]]
        # for i in range(5):
        #     x = h[i][0] + (self.Eps/2*np.sqrt(3)) * \
        #         np.cos((5*np.pi/6) + i*np.pi/3)
        #     y = h[i][1] + (self.Eps/2*np.sqrt(3)) * \
        #         np.sin((5*np.pi/6) + i*np.pi/3)
        #     h.append(np.round([x, y], 2))



        # h=[]
        # for i in range(1,7):
        #     angle_deg = 60 * i - 30
        #     angle_rad = np.pi / 180 * angle_deg
        #     x = p[0] + (math.sqrt(3)/2*self.Eps) * np.cos(angle_rad)
        #     y = p[1] + (math.sqrt(3)/2*self.Eps) * np.sin(angle_rad)
        #     # h.append(np.round([x, y], 2))
        #     h.append([x, y])



        h=[]
        x = p[0] + 0.75*self.Eps
        y = p[1] + (np.sqrt(3))/4*self.Eps
        h.append([x, y])

        x = p[0] 
        y = p[1] + (np.sqrt(3))/2*self.Eps
        h.append([x, y])
        
        x = p[0] - 0.75*self.Eps
        y = p[1] + (np.sqrt(3))/4*self.Eps
        h.append([x, y])
        
        x = p[0] - 0.75*self.Eps
        y = p[1] - (np.sqrt(3))/4*self.Eps
        h.append([x, y])
        
        x = p[0] 
        y = p[1] - (np.sqrt(3))/2*self.Eps
        h.append([x, y])
        
        x = p[0] + 0.75*self.Eps
        y = p[1] - (np.sqrt(3))/4*self.Eps
        h.append([x, y])
        
        
        
        return(np.array(h))
# ------------------------------

# ==============================
#d1 = np.random.normal(0, 10, (500,2))
#d2 = np.random.normal(20, 5, (250,2))
#data = np.concatenate((d1,d2))


#point = np.array([0,0])
#self.Eps = 3
#hex1 = ahex(point,self.Eps)
#n1 = hexn(point,self.Eps)

#n1hx = []
# for i in range(len(n1)):
#    n1hx.append(ahex(n1[i],self.Eps))
#n1hx = np.array(n1hx)

# plt.plot(data[:,0],data[:,1],".")
# for i in range(6):
# plt.plot(n1hx[i,:,0],n1hx[i,:,1],"m")
# plt.plot(hex1[:,0],hex1[:,1],"r")
# plt.show()

#poly = path.Path(hex1)

#inner_data = poly.contains_points(data)
#inner_data, = np.where(inner_data)
# print(inner_data)

#fig, ax = plt.subplots()
#patch = patches.PathPatch(poly, edgecolor="m",facecolor="none")
# ax.add_patch(patch)
# ax.set_xlim(min(data[:,0])-10,max(data[:,0])+10)
# ax.set_ylim(min(data[:,1])-10,max(data[:,1])+10)
# ax.plot(data[:,0],data[:,1],".")
# plt.show()
