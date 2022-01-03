
import os
from typing import Callable, Optional
from pandas.core.frame import DataFrame

from PIL import Image

from torchvision.datasets import VisionDataset

class CaptionDataset(VisionDataset):
    """Custom Caption Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, 
        root: str, 
        filename_caption:DataFrame, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None,
        is_train: bool = True
        ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)     

        filename_caption = filename_caption if is_train else (
            filename_caption.groupby('image')['caption'].
            apply(list).
            reset_index()
        )

        self.imgs = filename_caption["image"].to_list()
        self.captions = filename_caption["caption"].to_list()

    def __getitem__(self, index):

        img_path = os.path.join(self.root, self.imgs[index])

        return self.transforms(
            input=Image.open(img_path).convert('RGB'),
            target=self.captions[index]
        )

    def __len__(self):
        return len(self.imgs)