from algorithm.utility import Distance
            
class KNearestNeibourClassifier:
    def __init__(self, k=5):
        self.x = None
        self.y = None
        self.k = k
    
    def fit(self,x,y):
        self.x = x
        self.y = y

    def freq(self, values):
        res = dict()
        for i in values:
            try:
                res[i] = res[i]+1
            except:
                res[i] = 1
        return res
            
    def predict(self, test_data):
        distances = list()
        for i in range(len(self.x)):
            d = Distance(self.x[i],test_data)
            distances.append([self.y[i], d.ecludian()])
        distances.sort(key=lambda distances: distances[1])
        counts = list()
        for i in distances[:self.k]:
            counts.append(i[0])
        frequencies = self.freq(counts)
        res_d = -1
        res_val = None
        for i in frequencies.keys():
            if frequencies[i] > res_d:
                res_val = i
        return res_val


