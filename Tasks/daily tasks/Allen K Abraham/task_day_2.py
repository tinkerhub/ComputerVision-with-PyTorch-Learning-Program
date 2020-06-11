import numpy as np
import torch

def main():

  # Initialization of numpy arrays
  arrayOne = np.random.randint(10, size=(5, 3))
  arrayTwo = np.random.randint(5, size=(3, 4))

  # Converting numpy arrays to torch tensors
  tensorArrayOne = torch.from_numpy(arrayOne)
  tensorArrayTwo = torch.from_numpy(arrayTwo)

  # Multiplying two tensors
  tensorResult = torch.matmul(tensorArrayOne, tensorArrayTwo)

  print(tensorResult)


if __name__ == "__main__":
  main()