import numpy as np
import math 

def euclidean_distance(v1 : np.array, v2 : np.array):
    distance = np.sqrt(sum((a_i - b_i)**2 for a_i ,b_i in zip(v1,v2)))
    return distance

def cosine_similarity(v1 : np.ndarray, v2 : np.ndarray):
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    sim_cos = ( np.dot(v1 , v2) ) / (v1_norm * v2_norm)
    return sim_cos

class KNNClassifier:
    def __init__(self,k=3,metric="euclidean"):
        self.k = k
        self.metric =metric
        self.X = None
        self.y = None

    def fit(self, X , y):
        self.X = X
        self.y = y

    def predict(self, query):
        euc_dis = []
        cos_sims = []

        for X_i in self.X :
            # print(X_i)
            euc_distance = euclidean_distance(X_i , query)
            cos_similarity = cosine_similarity(X_i ,query)
            euc_dis.append(euc_distance)
            cos_sims.append(cos_similarity)

        if self.metric == "euclidean":
            indices = np.argsort(euc_dis)
        
        elif self.metric == "cosine":
            indices = np.argsort(cosine_similarity)
            indices = np.argsort(cos_sims)[::-1]

        else:
            indices = []

        # print(indices)
    
        top_k_indices = indices[:self.k]
        # print(top_k_indices)
        top_k_labels = self.y[top_k_indices]
        # print(top_k_labels)

        out =  np.bincount(top_k_labels).argmax()

        return out

X = np.array([[1, 1], [2, 2], [10, 10], [11, 11]]) 
y = np.array([0, 0, 1, 1])
query = np.array([3, 3])

knn = KNNClassifier()
knn.fit(X,y)
print(knn.predict(query))
