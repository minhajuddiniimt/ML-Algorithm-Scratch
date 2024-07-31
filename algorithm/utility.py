import numpy as np

class Base:
    def __init__(self):
        self.x = None
        self.y = None
    
class Activation:
    def __init__(self,value):
        self.value = value

    def sigmoid(self):
        return 1/(1+np.exp(-self.value))

class Metrics:
    @staticmethod
    def accuracy(y,y_preds):
        count=0
        for i in range(len(y)):
            if y[i]==y_preds[i]:
                count = count + 1
        return count / len(y)
    
class Distance:
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def ecludian(self):
        sum = 0
        for i in range(len(self.x)):
            sum = sum + (self.x[i]-self.y[i])**2
        return sum**0.5