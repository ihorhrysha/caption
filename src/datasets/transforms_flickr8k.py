import torchvision.transforms as T

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transforms = {

    'init': T.Compose([
        T.Resize(226), 
        T.RandomCrop(224),
        T.ToTensor(), 
        T.Normalize(MEAN, STD)
    ])

}
