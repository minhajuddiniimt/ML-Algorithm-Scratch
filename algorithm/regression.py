import numpy as np

class LinearRegression:
    def __self__(self):
        self.w = None
        self.x = None
        self.y = None
        self.verbose = -1
        self.validation_split = None

    def error(self,y,y_pred):
        err = (y-y_pred)**2
        err = np.sum(err)
        err = err/y.shape[0]
        err = err**(1/2)
        return err

    def fit(self,x,y, epochs=10000, learning_rate=0.01):
        self.x = np.append(x, np.ones((x.shape[0],1)), axis=1)
        self.y = y
        self.w = np.random.rand(x.shape[1]+1)
        print(self.w)
        for epoch in range(epochs):
            y_preds = list()
            for i in range(len(self.x)):
                y_pred = self.predict(self.x[i])
                y_preds.append(y_pred)
            print(self.y)
            print(y_preds)
            err = self.error(self.y,y_pred)
            print(epoch, " Error : ",err)
            w_new = self.w - 2*learning_rate*(np.dot((y_preds-y),self.x)/self.x.shape[0])
            print(self.w)
            if np.sum(self.w == w_new)==self.w.shape[0]:
                break
            else:
                self.w = w_new

            print("-------------------------------------------------------------------")
        print(self.w)
        
    def predict(self,x):
        pred = np.dot(x,self.w)
        return pred