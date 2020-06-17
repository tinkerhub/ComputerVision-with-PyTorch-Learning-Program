

from torchvision import  transforms
from PIL import Image

import torchvision.transforms.functional as F
import torch



transform = transforms.Compose([
transforms.Resize(255),
transforms.CenterCrop(224),
transforms.ColorJitter(brightness=1, contrast=1, saturation=0, hue=0),
transforms.RandomVerticalFlip(),
   transforms.ToTensor(),
      transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        ),
      

])


img=Image.open('image/index.jpeg')

img = transform(img)


a = F.to_pil_image(img)
a.show()
