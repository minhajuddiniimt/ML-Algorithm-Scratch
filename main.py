from algorithm.regression import LinearRegression
from sklearn.datasets import make_regression, make_blobs

from algorithm.KNearestNeighbour import KNearestNeibourClassifier
from algorithm.LogisticRegression import LogisticRegression
from algorithm.NaiveBayes import NaiveBayes


import numpy as np
import math

if __name__ == "__main__":
    # testing the linear regression 
    # x, y = make_regression(n_samples=5, n_features=2, noise=1, random_state=42)
    # model = LinearRegression()
    # model.fit(x,y)   

    # testing the knn
    x,y = make_blobs(random_state=42)
    knn = KNearestNeibourClassifier()
    knn.fit(x, y)
    print(knn.predict(x[0]+6.4))
    print(knn.predict(x[0]))

    # testing the logistic regression using only for binary classification.
    # x,y = make_blobs(centers=2, random_state=100)
    # logit = LogisticRegression()
    # logit.fit(x, y)
    # print(logit.predict(x[:5]))

    # x,y = make_blobs(centers=3, random_state=100)
    # bayes = NaiveBayes()
    # bayes.fit(x, y)
    # print(logit.predict(x[:5]))

    # x1 = [[np.random.randint(1,3), np.random.randint(1,3)] for i in range(50)]
    # x1 = np.array(x1)
    # y1 = ['a' for i in range(50)]

    # x2 = [[np.random.randint(4,6), np.random.randint(4,6)] for i in range(50)]
    # x2 = np.array(x2)
    # y2 = ['b' for i in range(50)]
    # x = np.vstack((x1,x2))
    # y = y1 + y2
    # y = np.array(y)
    # print(x)
    # print(y)

    # bayes = NaiveBayes()
    # bayes.fit(x, y)


    # bayes.predict(x[:5])



