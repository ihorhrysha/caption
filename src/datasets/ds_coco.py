import os
from typing import List

import torch

import nltk
from PIL import Image
from pycocotools.coco import COCO

from datasets.transforms_coco import transforms
from datasets.vocab import VocabularyBuilder, Vocabulary
import torch.utils.data as data
from constants import TRAIN,VAL
from datasets.collate import collate_fn

class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, caption_path, vocab_path, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            caption_path: coco annotation file path.
            vocab_path: path to vocab pkl
            transform: image transformer.
        """
        self.root = root
        self.coco: COCO = COCO(caption_path)
        self.ids = list(self.coco.anns.keys())
        self.vocab: Vocabulary = VocabularyBuilder.get(
            captions=[self.coco.anns[id]['caption'] for id in self.coco.anns.keys()],
            threshold=4,
            vocab_path=vocab_path,
            tokenizer=self.tokenizer
        )
        self.transform = transform
    
    @staticmethod
    def tokenizer(caption:str)-> List[str]:
        return nltk.tokenize.word_tokenize(str(caption).lower())

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""

        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        target = torch.Tensor([
            self.vocab(self.vocab.START),
            *[self.vocab(token) for token in self.tokenizer(caption)],
            self.vocab(self.vocab.END)
        ])
        return image, target

    def __len__(self):
        return len(self.ids)

class CocoDataProvider:
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

        path_ann = path_ann or os.path.join(path_data, 'annotations')
        path_vocab = path_vocab or os.path.join(path_data, 'vocab')

        transform_keys = transform_keys or {TRAIN: "init", VAL: "init"}
        batch_sizes = batch_sizes or {TRAIN: 64, VAL: 64}
        data_type_labels ={
            TRAIN: "train2014",
            VAL: "val2014"
        }

        self.dataset: dict[str, CocoDataset] = {}
        self.loader: dict[str, data.DataLoader] = {}
        for data_type in [TRAIN, VAL]:
            
            data_type_label = data_type_labels[data_type]
            caption_path = os.path.join(path_ann, f"captions_{data_type_label}.json")

            self.dataset[data_type] = CocoDataset(
                root=os.path.join(path_data, data_type_label),
                caption_path=caption_path,
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
