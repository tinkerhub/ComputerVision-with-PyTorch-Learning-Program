import torch
from PIL import Image
from torchvision import  transforms
import torchvision.transforms.functional as F

transform = transforms.Compose([
transforms.Resize(300),
transforms.RandomCrop(200),
transforms.ColorJitter(brightness=0.7, contrast=0.3, saturation=0.3, hue=0.3),
transforms.RandomRotation((-60,60), resample=False, expand=False, center=None, fill=None),
transforms.RandomHorizontalFlip(),
transforms.RandomVerticalFlip(),
   transforms.ToTensor(),
      transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        ),


])

path="image/flower.jpeg"
img=Image.open(path)

img = transform(img)

a = F.to_pil_image(img)
a.show()