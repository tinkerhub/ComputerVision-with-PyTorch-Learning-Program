import torch
import torch as nn

#creating a class as Net
class Net(nn.module):

  def __init__(self,input_layer,hidden_layer1,hidden_layer2,output_layer):
  
    super(Net,self).__init__() 
    self.t1=nn.Linear(input_size,hiden_layer1)
    
    self.f1=nn.sigmoid()
  
    self.t2=nn.Linear(hidden_layer1,hiddenl_layer2)
    
    self.f2=nn.sigmoid()
    
    self.t3=nn.linear(hidden_layer2,output_layer)
    
    
  def forward(self,x):
    x=self.t1(x)
    x=self.f1(x)
    x=self.t2(x)
    x=self.f2(x)
    x=self.t3(x)
    
   return x
    
    
