import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:

    def fit(self, X_train, Y_train):
        self.X_train = [j for i in X_train for j in i]
        self.Y_train = Y_train
        self.model = np.polyfit(self.X_train,self.Y_train, 1)

        ## Simple model showing the relationship between x values and y values
        # plt.scatter(self.X_train,self.Y_train)
        # plt.title('X vs. Y')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.show()

    def predict(self, X_test):
        predict = np.poly1d(self.model)
        Y_test = int(predict(X_test))
        print('For X =' , X_test , ', Y =', Y_test)

## Trying the LinearRegression class for the given data
# x = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
# y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# X = [j for i in x for j in i]

# model = LinearRegression()
# model.fit(X,y)

# model.predict(10)
# model.preduct(11)
