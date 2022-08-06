from NeighbourGridQuery import *
from NeighbourHex import *
from scipy.spatial import distance
import random
import json
import time


class GDCF:
    def __init__(self, CoreGrids, coreobjects, dim, b, Minpts, Eps):
        self.Core_G = CoreGrids
        self.dimention = dim
        self.HGBLst = b
        self.Minpts = Minpts
        self.Eps = Eps
        self.Core_Objects = coreobjects
    def BuildGDCF(self, mode,HS, DataGrids, Data, NonEmptyGrids):
        # -----------------------------------------LDF---------------------------------------

        if mode == "LDF":
            print("ldf")
            # L=[]
            # Q=[]
            # for i in range(len(self.Core_Grids)):
            #     if self.Core_Grids[i]!=[]:
            #         L.append([len(DataGrids[i]),i])
            # L=sorted(L)
            # for j in range(len(L)):
            #     Q.append(self.Core_Grids[L[j][1]])
            L = []
            Q = []
            # print(f"lenght datagris : {len(DataGrids)}")nnn
            # print(f"lenght self.Core_Grids : {len(self.Core_G)}")
            for g, grid in enumerate(self.Core_G):
                if grid != []:
                    # print(f"g : {g} grid : {grid}")
                    L.append([len(DataGrids[g]), g])
            L = sorted(L)
            for j in range(len(L)):
                Q.append(self.Core_G[L[j][1]])
                # print(f"order ccore grids Q : {Q}")nnn

        # ----------------------------------Random-------------------------------------------
        def myFunc():
            return 0.2
        if mode == "Random":
            print("random")
            L = []
            Q = []
            for i in range(len(self.Core_G)):
                L.append([len(self.Core_G[i]), i])
            random.shuffle(L,myFunc)
            for j in range(len(L)):
                Q.append(self.Core_G[L[j][1]])

        # ---------------------------------LNN-------------------------------------------------
        if mode == "LNN":
            print("lnn")
            Neighbours = []
            for g in self.Core_G:
                if g != []:
                    G1 = NeighbourGridQuery(g, self.dimention, self.HGBLst)
                    N = G1.NeighbourGrid(NonEmptyGrids)
                    Neighbours.append([g, N])
            Q = list(sorted(Neighbours, reverse=True, key=sort))
            # print(f"g {g}")
            print(f"Q {Q}")
                    

        # ----------------------------GDFC-------------------------------------------------------
        # print(f"Q(Core Grids) : {Q}")
        X = 1
        Forest = [[None]]
        A = []
        # h = 1
        # h2 = 1
        offset = -1
        for g in Q:
        # for gi, g in enumerate(Q, start=(offset+1)):
            # # read
            # with open('/content/drive/MyDrive/Colab Notebooks/saveobject.json', 'r') as openfile:
            #     # Reading from json file
            #     save_object = json.load(openfile)
            #     openfile.close()
            # # update
            # save_object['count_g'] = gi
            # # write
            # with open('/content/drive/MyDrive/Colab Notebooks/saveobject.json', 'w') as openfile:
            #     Saveobj = json.dump(save_object, openfile)
            #     openfile.close()
            if mode == "LDF" or mode == "Random":
                if HS == "Square":
                    G1 = NeighbourGridQuery(g, self.dimention, self.HGBLst)  # LDF
                    G = G1.NeighbourGrid(NonEmptyGrids)  # LDF


                if HS == "Hex":
                    G1 = NeighbourHex(g, self.dimention, self.HGBLst)  # LDF
                    G = G1.NeighbourGrid(NonEmptyGrids)  # LDF
                    # print(f"Neighbor g : {G}{g}")nnn
                
                g = [g, G]  # LDF"""

            # # read
            # with open('/content/drive/MyDrive/Colab Notebooks/saveobject.json', 'r') as openfile:
            #     # Reading from json file
            #     save_object = json.load(openfile)
            #     openfile.close()
            # # update
            # save_object['neighbour'] = G
            # # write
            # with open('/content/drive/MyDrive/Colab Notebooks/saveobject.json', 'w') as openfile:
            #     Saveobj = json.dump(save_object, openfile)
            #     openfile.close()
            
            A = [g[0]]
            if any(g[0] in sublist for sublist in Forest) == False:  # نباشد P عضو جنگل  g
                Tree = [X, g[0]]
                Forest.append(Tree)
                X += 1
                # print(f"add new tree to Forest(g dont be in Forest) : {Forest}")nnn

            offset2 = -1
            for gprim in g[1]:
                # print(f"gprim : {gprim}")nnn
            # for gprim_count, gprim in enumerate(g[1], start=(offset2+1)):
                # # read
                # with open('/content/drive/MyDrive/Colab Notebooks/saveobject.json', 'r') as openfile:
                #     # Reading from json file
                #     save_object = json.load(openfile)
                #     openfile.close()
                # # update
                # save_object['count_gprim'] = gprim_count
                # # write
                # with open('/content/drive/MyDrive/Colab Notebooks/saveobject.json', 'w') as openfile:
                #     Saveobj = json.dump(save_object, openfile)
                #     openfile.close()

                Root_g, _ = findRoot(Forest, g[0])
                Root_gprim, _ = findRoot(Forest, gprim)
                # indx_g=self.C.index(g[0])new
                indx_g = NonEmptyGrids.index(g[0])
                indx_gprim = NonEmptyGrids.index(gprim)
                if Root_g == Root_gprim:
                    # print(f"g , gprim root : {Root_g}{Root_gprim}")nnn
                    continue
                
                # if g in self.Core_G and gprim in self.Core_G:
                if mergability(self, indx_g, indx_gprim, DataGrids, Data) == True:
                    if any(gprim in sublist for sublist in Forest) == False:
                        Forest[Root_g].append(gprim)  # new
                        # print(f"g , gprim are mergable and gprim add to forest : {Forest}")nnn
                    else:
                        A.append(gprim)
                        # print(f"g , gprim are not mergable and gprim add to A : {A}")nnn



# ***********set parents of all roots of cluster numbers in set A tolrc(A)***********************************
            Clusters_roots = []  # number of root for all of clusters
            for grid in A:
                Root, _ = findRoot(Forest, grid)
                if Root != None and (Root not in Clusters_roots):
                    Clusters_roots.append(Root)

            Irc = min(Clusters_roots)
            Irc_index = Clusters_roots.index(Irc)

            # for grid in A:
            #    _,RootIndex=findRoot(Forest,grid)
            #    if RootIndex!=None and Forest[RootIndex][0]!=Irc:
            #        indx=Forest[RootIndex].index(grid)#???
            #        for z in Forest:
            #            if z[0]==Irc:
            #                z.append(Forest[RootIndex][indx])
            #                del Forest[RootIndex][indx]

            # if len(Clusters_roots)>1:
            #    _,RootIndexIrc=findRoot(Forest,A[Irc_index])
            #    for grid in A:
            #        _,RootIndex=findRoot(Forest,grid)
            #        if RootIndex!=None and Forest[RootIndex][0]!=Irc:
            #            for z in Forest:
            #                if z[0]==Irc:
            #                    length=len(Forest[RootIndex])-1
            #                    while(length+1!=1):
            #                        z.append(Forest[RootIndex][length])
            #                        del Forest[RootIndex][length]
            #                        length-=1

            if len(Clusters_roots) > 1:
                _, RootIndexIrc = findRoot(Forest, A[Irc_index])
                for grid in A:
                    _, RootIndex = findRoot(Forest, grid)
                    if RootIndex != None and Forest[RootIndex][0] != Irc:
                        for i in range(1, len(Forest[RootIndex])):
                            Forest[RootIndexIrc].append(Forest[RootIndex][i])
                        Forest[RootIndex].clear()
            #  # read
            # with open('/content/drive/MyDrive/Colab Notebooks/saveobject.json', 'r') as openfile:
            #     # Reading from json file
            #     save_object = json.load(openfile)
            #     openfile.close()
            # # update
            # save_object['forest'] = Forest
            # save_object['cluster_number'] = X
            # save_object['A'] = A
            # # write
            # with open('/content/drive/MyDrive/Colab Notebooks/saveobject.json', 'w') as openfile:
            #     Saveobj = json.dump(save_object, openfile)
            #     openfile.close()

        Noise = []
        NotCoreGrids = []
        NotCoreGrids = [x for x in NonEmptyGrids if not x in self.Core_G]
       # list(set(NonEmptyGrids)-set(self.C))
        for grid in NotCoreGrids:
            if any(grid in sublist for sublist in Forest):
                grid[2] = 'Border Points'
            else:
                Noise.append(grid)
        # print(f"Noise : {Noise}")nnn
                # i=NonEmptyGrids.index(grid)
                # for n in DataGrids[i]:
                #    Noise.append(n)
        # Noise=[]
        # t=[]
        # h=0
        # for grid in self.C:
        #    if grid!=[]:
        #        h+=1

        # for i in range(len(Forest)):
        #    for j in range(1,len(Forest[i])):
        #        if Forest[i][j] in NonEmptyGrids:
        #            t.append(Forest[i][j])
        #        else:
        #            Noise.append(Forest[i][j])

        Clusters = []
        c = 0
        for i in range(len(Forest)):
            if len(Forest[i]) > 1:
                Clusters.append([])
                for j in range(1, len(Forest[i])):
                    indx = NonEmptyGrids.index(Forest[i][j])
                    for h in range(len(DataGrids[indx])):
                        Clusters[c].append(DataGrids[indx][h])
                c += 1
        # print(f"Clusters : {Clusters}")nnn
        Cluster_Num = np.zeros([len(Data), self.dimention+1])
        for i in range(len(Clusters)):
            for j in range(len(Clusters[i])):
                for k in range(self.dimention+1):
                    # self.Minpts
                    if k == self.dimention and len(Clusters[i]) < self.Minpts:
                        Cluster_Num[Clusters[i][j]][k] = -1
                    elif k == self.dimention:
                        Cluster_Num[Clusters[i][j]][k] = i+1
                    else:
                        Cluster_Num[Clusters[i][j]][k] = Data[Clusters[i][j]][k]
        # print(f"Cluster_Num : {Cluster_Num}")nnn
        return Cluster_Num
        print("m")


def mergability(self, ind_g, ind_gprim, DataGrids, Data):
    # m=[[],[]]
    for data in range(len(DataGrids[ind_g])):
        for dataNeighbourGrid in range(len(DataGrids[ind_gprim])):
            dist = distance.euclidean(
                Data[DataGrids[ind_g][data]], Data[DataGrids[ind_gprim][dataNeighbourGrid]])  # Calculate euclidean
            if dist <= self.Eps and (DataGrids[ind_g][data] in self.Core_Objects and DataGrids[ind_gprim][dataNeighbourGrid] in self.Core_Objects):
                return True

        # if data[0]==g[0]:
        #    m[0].append([Data[0][data],Data[1][data]])#coordinates of objects in grid g
        # if data[0]==gprim[0]:
        #    m[1].append([Data[0][data],Data[1][data]])#coordinates of objects in grid gprim

        # for i in DataGrids[g]:
        #    for j in DataGrids[gprim]:
        #        dist=distance.euclidean(i,j)#Calculate euclidean
        #        if dist<=3:
        #            return True
    return False


def findRoot(Forest, leaf):
    flag = False
    for Sublist_Forest in Forest:
        if leaf in Sublist_Forest:
            flag = True
            break
    if flag:
        return Forest[Forest.index(Sublist_Forest)][0], Forest.index(Sublist_Forest)
    else:
        return None, None
    # for i in range(len(jungle)):
    #    if leaf in jungle[i]:
    #        return jungle[i][0],jungle[jungle.index(jungle[i])][0]


def sort(e):
    return len(e[1])
