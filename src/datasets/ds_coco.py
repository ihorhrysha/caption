import os

from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Subset
import pandas as pd

from datasets.transforms import get_transforms, get_tokenizer
from datasets.vocab import VocabularyBuilder
from constants import TRAIN,VAL
from datasets.collate import collate_fn, collate_val_fn
from datasets.caption_dataset import CaptionDataset

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
                 transform_config=None):

        data_type_labels ={
            TRAIN: "train2014",
            VAL: "val2014"
        }

        path_ann = path_ann or os.path.join(path_data, 'annotations')
        path_vocab = path_vocab or path_data

        batch_sizes = batch_sizes or {TRAIN: 64, VAL: 64}
        transform_config = transform_config or {TRAIN: 'coco', VAL: 'coco_init', 'tokenizer':'nltk', 'threshold':4}

        self.dataset: dict[str, CaptionDataset] = {}
        self.loader: dict[str, DataLoader] = {}
        
        filename_caption = {
            TRAIN: self.coco2pd(caption_path = os.path.join(path_ann, f"captions_{data_type_labels[TRAIN]}.json")), 
            VAL: self.coco2pd(caption_path = os.path.join(path_ann, f"captions_{data_type_labels[VAL]}.json"))
        }
        
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
                root=os.path.join(path_data, data_type_labels[data_type]),
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

    @staticmethod
    def coco2pd(caption_path:str):
        coco: COCO = COCO(caption_path)

        img_id2filename = {img_id: img_info['file_name'] for img_id, img_info in coco.imgs.items()}

        captions = []
        images = []
 
        for _, caption_info in coco.anns.items():
            images.append(img_id2filename[caption_info['image_id']])
            captions.append(caption_info['caption'])

        return pd.DataFrame({"image":images, "caption":captions})
