import os

import torch
# import torchvision.datasets as dset

import nltk
from PIL import Image
from pycocotools.coco import COCO

from datasets.transforms_coco import transforms
from datasets.vocab_coco import VocabularyBuilder
import torch.utils.data as data
from constants import TRAIN,VAL

class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, caption_path, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            caption_path: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(caption_path)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

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
            
            vocab = VocabularyBuilder.get(
                caption_path=caption_path,
                threshold=4,
                vocab_path=os.path.join(path_vocab,f'{data_type}.pkl')
            )

            self.dataset[data_type] = CocoDataset(
                root=os.path.join(path_data, data_type_label),
                caption_path=caption_path,
                vocab=vocab,
                transform=transforms[transform_keys[data_type]]
            )

            self.loader[data_type] = data.DataLoader(
                dataset=self.dataset[data_type],
                batch_size=batch_sizes[data_type],
                shuffle=(data_type == TRAIN),
                num_workers=num_workers,
                collate_fn=collate_fn
            )
