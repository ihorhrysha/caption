 
import nltk
import pickle
import argparse
import os
from collections import Counter
from pycocotools.coco import COCO


class Vocabulary:
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

class VocabularyBuilder:

    @staticmethod
    def get(caption_path, threshold, vocab_path):
        if os.path.exists(vocab_path):
            print('Load vocab from cache...')
            return VocabularyBuilder.load(vocab_path)
        else:
            print('Creating new vocab index...')
            vocab = VocabularyBuilder.build(caption_path, threshold)
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
    def build(caption_path, threshold):
        """Build a simple vocabulary wrapper."""
        coco = COCO(caption_path)
        counter = Counter()
        ids = coco.anns.keys()
        for i, id in enumerate(ids):
            caption = str(coco.anns[id]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

            if (i+1) % 1000 == 0:
                print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

        # If the word frequency is less than 'threshold', then the word is discarded.
        words = [word for word, cnt in counter.items() if cnt >= threshold]

        # Create a vocab wrapper and add some special tokens.
        vocab = Vocabulary()
        vocab.add_word('<pad>')
        vocab.add_word('<start>')
        vocab.add_word('<end>')
        vocab.add_word('<unk>')

        # Add the words to the vocabulary.
        for i, word in enumerate(words):
            vocab.add_word(word)
        return vocab

def main(args):
    vocab = VocabularyBuilder.build(caption_path=args.caption_path, threshold=args.threshold)
    VocabularyBuilder.save(vocab, args.vocab_path)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(args.vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='data/coco/annotations/captions_train2014.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./materials/vocab/coco.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)