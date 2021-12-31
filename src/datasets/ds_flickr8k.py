import os
from pandas.core.frame import DataFrame

import torch

import nltk
from PIL import Image
import pandas as pd

from datasets.transforms_flickr8k import transforms
from datasets.vocab import VocabularyBuilder, Vocabulary
import torch.utils.data as data
from constants import TRAIN,VAL
from datasets.collate import collate_fn



class Flickr8kDataset(data.Dataset):
    """Flickr8kDataset Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, filename_caption:DataFrame, vocab_path, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            captions: coco annotation file path.
            vocab_path: path to vocab pkl
            transform: image transformer.
        """
        self.root = root
        
        self.imgs = filename_caption["image"].to_list()
        self.captions = filename_caption["caption"].to_list()

        self.vocab: Vocabulary = VocabularyBuilder.get(
            captions=self.captions,
            threshold=5, # TODO also good candidate for experiments
            vocab_path=vocab_path,
            tokenizer=self.tokenizer
        )
        self.transform = transform
    
    @staticmethod
    def tokenizer(caption:str)-> list[str]:
        # TODO try spacy[en] and make toketizer configurable
        return nltk.tokenize.word_tokenize(str(caption).lower())

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""

        img_path = os.path.join(self.root, self.imgs[index])
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        target = torch.Tensor([
            self.vocab(self.vocab.START),
            *[self.vocab(token) for token in self.tokenizer(self.captions[index])],
            self.vocab(self.vocab.END)
        ])
        return image, target

    def __len__(self):
        return len(self.imgs)

class Flickr8kProvider:
    """
    Class to manage COCO caption train and val datasets
    """

    def __init__(self,
                 path_data,
                 path_ann=None,
                 path_vocab=None,
                 num_workers=4,
                 batch_sizes=None,
                 download=False,
                 tiny=False,
                 transform_keys=None):

        path_ann = path_ann or os.path.join(path_data, 'captions.txt')
        path_vocab = path_vocab or os.path.join(path_data, 'vocab')

        transform_keys = transform_keys or {TRAIN: "init", VAL: "init"}
        batch_sizes = batch_sizes or {TRAIN: 64, VAL: 64}
                
        # as we have only one file here is a simple way to split our dataset
        df_ann = pd.read_csv(path_ann)
        val_img_count = 500
        val_cutoff = len(df_ann)-val_img_count*5
        filename_caption = {TRAIN: df_ann[:val_cutoff], VAL: df_ann[val_cutoff:]}
        
        self.dataset: dict[str, Flickr8kDataset] = {}
        self.loader: dict[str, data.DataLoader] = {}
        for data_type in [TRAIN, VAL]:

            self.dataset[data_type] = Flickr8kDataset(
                root=os.path.join(path_data, 'Images'),
                filename_caption=filename_caption[data_type],
                vocab_path=os.path.join(path_vocab,f'{data_type}.pkl'),
                transform=transforms[transform_keys[data_type]]
            )

            self.loader[data_type] = data.DataLoader(
                dataset=self.dataset[data_type],
                batch_size=batch_sizes[data_type],
                shuffle=(data_type == TRAIN),
                num_workers=num_workers,
                collate_fn=collate_fn
            )
