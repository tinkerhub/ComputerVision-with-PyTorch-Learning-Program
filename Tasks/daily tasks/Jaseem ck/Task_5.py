#Task 5
#Run these codes in colab

#--------------------------------------------------------------------------------
import os
os.chdir('/content/drive/My Drive/Colab Notebooks')
print(os.getcwd())
#--------------------------------------------------------------------------------

#upload an image in the following directory in your google drive
path = "/content/drive/My Drive/Colab Notebooks/photo.jpg"

#--------------------------------------------------------------------------------

import torch
from PIL import Image
from torchvision import  transforms
import torchvision.transforms.functional as F

transform = transforms.Compose([
transforms.Resize(300),
transforms.CenterCrop(200),
transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
transforms.RandomRotation((-60,60), resample=False, expand=False, center=None, fill=None),
transforms.RandomHorizontalFlip(),
transforms.RandomVerticalFlip(),
   transforms.ToTensor(),
      transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        ),


])


img=Image.open(path)

img = transform(img)

a = F.to_pil_image(img)
b = F.to_grayscale(a, num_output_channels=1)

#--------------------------------------------------------------------------------

#a.show()
a

#--------------------------------------------------------------------------------

#b.show()
b
