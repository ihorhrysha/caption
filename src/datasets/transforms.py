from typing import Callable, Tuple
from torch.functional import Tensor
import torchvision.transforms as T

from datasets.vocab import Vocabulary

def tokenize_nltk(caption:str)-> list[str]:
    import nltk
    return nltk.tokenize.word_tokenize(str.lower(caption))

def caption_to_tensor(tokens: list[str]) -> Tensor:
    return Tensor(tokens)

def add_start_end(tokens: list[str]) -> list[str]:
    return [Vocabulary.START] + tokens + [Vocabulary.END]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

input = {

    'flickr8k_init':T.Compose([
        T.Resize(226), 
        T.CenterCrop(224),
        T.ToTensor(), 
        T.Normalize(MEAN, STD)
    ]),

    'flickr8k':T.Compose([
        T.Resize(226), 
        T.RandomCrop(224),
        T.ToTensor(), 
        T.Normalize(MEAN, STD)
    ]),

    'coco': T.Compose([
        T.Resize(size=(256,256),antialias=True),
        T.RandomCrop(244),
        T.RandomHorizontalFlip(), 
        T.ToTensor(), 
        T.Normalize(MEAN, STD)
    ])

}

tokenizer = {
    'nltk': tokenize_nltk
}

def get_tokenizer(tokenizer_key:str)->Callable:
    return tokenizer[tokenizer_key]

def get_transforms(key:str, tokenizer_key:str, vocab:Callable) -> Tuple[Callable, Callable]:
    transform = input[key]
    tokenizer = get_tokenizer(tokenizer_key)
    transform_target = T.Compose([
        tokenizer,
        add_start_end,
        vocab,
        caption_to_tensor,
    ])
    return transform, transform_target
