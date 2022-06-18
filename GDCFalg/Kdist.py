from scipy.spatial import distance

class Kdist:
    """K-distance diagram"""
    def __init__(self,X,K):
        self.Data=X
        self.k=K
    def calculate_kn_distance(self):
        kn_distance = []
        for i in range(len(self.Data[0])):
            eucl_dist = []
            for j in range(len(self.Data[0])):
                if i!=j:
                    eucl_dist.append(distance.euclidean([self.Data[0][i],self.Data[1][i]],[self.Data[0][j],self.Data[1][j]]))
            eucl_dist.sort()
            kn_distance.append(eucl_dist[self.k])

        return kn_distance

