from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
import numpy as np
class Evaluation:
    def __init__(self,True_label,Predict_label):
        self.y_true=True_label
        self.y_pred=Predict_label

    def adjusted_rand(self):
        R=adjusted_rand_score(True_label, Pred_label)
        return R
    

    def purity_score(self):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(self.y_true,self.y_pred)
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 