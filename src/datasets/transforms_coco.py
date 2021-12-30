import torchvision.transforms as t
from torchvision.transforms import InterpolationMode

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transforms = {

    'init': t.Compose([
        t.Resize(size=(256,256),antialias=True),
        t.RandomCrop(244),
        t.RandomHorizontalFlip(), 
        t.ToTensor(), 
        t.Normalize(MEAN, STD)
    ])

}
