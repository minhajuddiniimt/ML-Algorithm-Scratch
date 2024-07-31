import numpy as np
from collections import Counter
class NaiveBayes:
    def __init__(self):
        self.x = None
        self.y = None
        self.prob = None
        self.unique = None
        self.target_prob = None
    
    def splitData(self, tg):
        temp = list()
        for i in range(len(self.y)):
            if self.y[i] == tg:
                temp.append(list(self.x[i]))
        temp = np.array(temp)
        return temp
    
    def cal_prob(self, sample):
        prob = dict(Counter(sample).items())
        for i in prob.keys():
            prob[i] = prob[i]/len(sample)
        return prob
    
    def cal_unique(self):
        self.unique = dict()
        for i in range(self.x.shape[1]):
            # self.unique[i] = set(self.x[:,i])
            self.unique[i] = self.cal_prob(self.x[:,i])
        print(self.unique)
        
    def handleColumn(self, tg):
        sample_data = self.splitData(tg)
        result = dict()
        for i in range(sample_data.shape[1]):
            result[i] = self.cal_prob(sample_data[:,i])
            keys = self.unique[i].keys()
            for key in keys:
                if result[i].get(key):
                    pass
                else:
                    result[i][key] = 0
        return result

        
    def fit(self, x, y):
        self.x = x
        self.y = y
        self.target_prob = self.cal_prob(self.y)
        self.cal_unique()
        # print(self.unique)
        self.prob = [ {i: self.handleColumn(i)} for i in self.target_prob.keys()]
        print(self.prob)

    def predict_s(self, data):
        res = list()
        for i,j in zip(self.prob, self.target_prob):
            p_x = self.target_prob[j]
            p_x_j = i
            print(p_x)
            print(j)
            
        print("------------")
            # p_x = self.target_prob[i]
            # print(p_x)
            # p = 1
            # for j in data:
            #     p = p * self.prob[i][self.target_prob.]
                




    def predict(self, data):
        pred = []
        for row in data:
            pred.append(self.predict_s(row))
        return pred



            