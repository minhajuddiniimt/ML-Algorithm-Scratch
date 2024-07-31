import numpy as np
from algorithm.utility import Activation, Metrics


class LogisticRegression:
    def __init__(self):
        self.x = None
        self.y = None
        self.w = None
       
    
    def fit(self, x, y, epochs=100, learning_rate=0.01):
        self.x = np.append(x, np.ones((x.shape[0],1)), axis=1)
        self.y = y
        self.w = np.random.rand(x.shape[1]+1)

        for epoch in range(epochs):
            y_preds = list()
            for i in range(len(self.x)):
                y_pred = self.predict_(self.x[i])
                y_preds.append(y_pred)
            w_new = self.w - learning_rate*(np.dot((y_preds-y),self.x)/self.x.shape[0])
            print(self.w)
            if np.sum(self.w == w_new)==self.w.shape[0]:
                break
            else:
                self.w = w_new
        print(self.y)
        print(y_preds)
        print("Accuracy : ", Metrics.accuracy(y,y_preds))


    def predict_(self,x):
        pred = np.dot(x,self.w)
        act = Activation(pred)
        return 1 if act.sigmoid() >= 0.5 else 0
    
    def predict(self,x):
        x = np.append(x, np.ones((x.shape[0],1)), axis=1)
        res = [self.predict_(i) for i in x]
        return res