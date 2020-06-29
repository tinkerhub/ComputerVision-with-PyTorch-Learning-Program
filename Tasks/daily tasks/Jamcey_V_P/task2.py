import torch
import torch.nn as nn

class neural_network(nn.Module):
  def __init__(self, ):
    super(neural_network, self).__init__()

    self.input_layer = 4                  # Number of input units
    self.hidden_layer1 = 5                # Number of hidden units
    self.hidden_layer2 = 3                # Number of hidden units  
    self.output_layer = 1                # Number of output units 
  
    # Weights 
    W1 = torch.randn(self.input_layer, self.hidden_layer1)  
    W2 = torch.randn(self.hidden_layer1, self.hidden_layer2) 
    W3 = torch.randn(self.hidden_layer2, self.output_layer) 

  
    #  bias 
    B1 = torch.randn((1, self.hidden_layer1)) 
    B2 = torch.randn((1,self.hidden_layer2))
    B3 = torch.randn((1,self.output_layer))

    def forward(self, X):
      z1 = torch.mm(X, w1) + b1
      Relu = nn.ReLU()
      a1 = Relu(z1)
      z2 = torch.mm(X, w2) + b2
      Relu = nn.ReLU()
      a2 = Relu(z2)
      z3 = torch.mm(X, w3) + b3
      Sigmoid = nn.Sigmoid()
      Result = Sigmoid(z3)
      return Result
