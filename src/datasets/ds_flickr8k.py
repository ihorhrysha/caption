import os
from typing import Callable, Optional
from pandas.core.frame import DataFrame

import torch

from PIL import Image
import pandas as pd

from datasets.transforms import get_transforms, get_tokenizer
from datasets.vocab import VocabularyBuilder, Vocabulary
# import torch.utils.data as data
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader, Subset
from constants import TRAIN,VAL
from datasets.collate import collate_fn, collate_val_fn

class CaptionDataset(VisionDataset):
    """Flickr8kDataset Custom Dataset compatible with torch.utils.data.DataLoader."""

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

class Flickr8kProvider:
    """
    Class to manage Flickr8k caption train and val datasets
    """

    def __init__(self,
                 path_data,
                 path_ann=None,
                 path_vocab=None,
                 num_workers=4,
                 batch_sizes=None,
                 download=False,
                 tiny=False,
                 transform_config=None):

        path_ann = path_ann or os.path.join(path_data, 'captions.txt')
        path_vocab = path_vocab or path_data

        batch_sizes = batch_sizes or {TRAIN: 64, VAL: 64}
        transform_config = transform_config or {TRAIN: 'flickr8k_init', VAL: 'flickr8k_init', 'tokenizer':'nltk', 'threshold':4}
                
        # as we have only one file here is a simple way to split our dataset
        df_ann = pd.read_csv(path_ann)
        val_img_count = 500
        val_cutoff = len(df_ann)-val_img_count*5
        filename_caption = {TRAIN: df_ann[:val_cutoff], VAL: df_ann[val_cutoff:]}
        
        self.dataset: dict[str, VisionDataset] = {}
        self.loader: dict[str, DataLoader] = {}
        
        # we need only one vocab for train dataset, validation should be done with train vocab,
        # it is better to store vocab on provider level
        self.vocab = VocabularyBuilder.get(
            captions=filename_caption[TRAIN]["caption"].to_list(),
            threshold=transform_config['threshold'],
            vocab_path=os.path.join(path_vocab,f'vocab.pkl'),
            tokenizer=get_tokenizer(transform_config['tokenizer'])
        )    

        for data_type in [TRAIN, VAL]:
            
            transform, target_transform = get_transforms(
                key=transform_config[data_type],
                tokenizer_key=transform_config['tokenizer'],
                vocab=self.vocab,
                is_train=(data_type == TRAIN)
            )

            self.dataset[data_type] = CaptionDataset(
                root=os.path.join(path_data, 'Images'),
                filename_caption=filename_caption[data_type],
                transform=transform,
                target_transform=target_transform,
                is_train=(data_type == TRAIN)
            )

            self.loader[data_type] = DataLoader(
                dataset=self.dataset[data_type],
                batch_size=batch_sizes[data_type],
                shuffle=(data_type == TRAIN),
                num_workers=num_workers,
                collate_fn=collate_fn if (data_type == TRAIN) else collate_val_fn
            )

        if tiny:
            for data_type in [TRAIN, VAL]:
                self.dataset[data_type] = Subset(self.dataset[data_type], range(batch_sizes[data_type]))
                self.loader[data_type] = DataLoader(
                    self.dataset[data_type], 
                    batch_size=batch_sizes[data_type],
                    collate_fn=collate_fn if (data_type == TRAIN) else collate_val_fn
                )