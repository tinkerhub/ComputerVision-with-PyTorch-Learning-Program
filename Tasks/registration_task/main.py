"""
Implement the linear regression model using python and numpy in the following class.
The method fit() should take inputs like,
x = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
"""

import numpy as np

class LinearRegression(object):
    def __init__(self, mx=None, my=None, b_0=None, b_1=None):
        self.mx=mx
        self.my=my
        self.b_0=b_0
        self.b_1=b_1
    def fit(self,_input, _output):
        n=np.size(_output)
        self.mx=np.mean(_input)
        self.my=np.mean(_output)
        sum1,sum2=0,0
        for i in range(n):
            sum1+=_input[i][0]*_output[i]
            sum2+=_input[i][0]*_input[i][0]
        SS_xy =sum1-n*self.my*self.mx 
        SS_xx =sum2-n*self.mx*self.mx
        self.b_1 = SS_xy / SS_xx 
        self.b_0 = self.my - self.b_1*self.mx
        print(self.b_0)
        print(self.b_1)
        print(self.my)
        print(self.mx)
        print(SS_xx)
        print(SS_xy)
        print(sum1)
        print(sum2)
      
      
    def predict(self,_input):
        n=np.size(_input)
        _output=[]
        for i in range(n):
            a=self.b_0+(self.b_1*_input[i][0])
            _output.append(a)
        return _output
