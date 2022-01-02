 
from typing import Iterable, Union
import pickle
import os
from collections import Counter


class Vocabulary:
    START = '<start>'
    END = '<end>'
    PAD = '<pad>'
    UNK = '<unk>'

    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        self.add_word(self.PAD)
        self.add_word(self.START)
        self.add_word(self.END)
        self.add_word(self.UNK)

    def add_word(self, word)->None:
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def _call_one(self, word_or_index:Union[int,str], return_index:bool=True) -> Union[str,int]:
        if return_index:
            return self.word2idx.get(word_or_index, self.word2idx[self.UNK])
        else:
            # return word
            return self.idx2word.get(word_or_index, self.UNK)

    def __call__(self, word_or_index:Union[int,str], return_index:bool=True) -> Union[str,int]:
        if isinstance(word_or_index, list):
            return [self._call_one(word_or_index=one, return_index=return_index) for one in word_or_index]
        else:
            return self._call_one(word_or_index=word_or_index, return_index=return_index)
            

    def __len__(self):
        return len(self.word2idx)


class VocabularyBuilder:

    @staticmethod
    def get(captions, threshold, vocab_path, tokenizer) -> Vocabulary:
        if os.path.exists(vocab_path):
            print('Load vocab from cache...')
            return VocabularyBuilder.load(vocab_path)
        else:
            print('Creating new vocab index...')
            vocab = VocabularyBuilder.build(captions, threshold, tokenizer=tokenizer)
            VocabularyBuilder.save(vocab, vocab_path)
            return vocab

    @staticmethod
    def load(vocab_path):
        with open(vocab_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def save(vocab, vocab_path):
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)

    @staticmethod
    def build(captions, threshold, tokenizer):
        """Tokenize and build vocab"""

        counter = Counter()
        
        for i, caption in enumerate(captions):
            
            tokens = tokenizer(caption)
            counter.update(tokens)

            if (i+1) % 1000 == 0:
                print("[{}/{}] Tokenized the captions.".format(i+1, len(captions)))

        # If the word frequency is less than 'threshold', then the word is discarded.
        words = [word for word, cnt in counter.items() if cnt >= threshold]

        # Create a vocab wrapper and add some special tokens.
        vocab = Vocabulary()

        # Add the words to the vocabulary.
        for word in words:
            vocab.add_word(word)
        return vocab