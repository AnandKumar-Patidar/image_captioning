from collections import defaultdict
from pycocotools.coco import COCO
import nltk
import argparse
import os
import pickle

class Vocabulary:
    def __init__(self, threshold):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.threshold = threshold

        # Special Tokesn
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx+=1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.word2idx)

def build_vocab(ann_file, threshold=5):
    from pycocotools.coco import COCO
    coco = COCO(ann_file)
    counter = defaultdict(int)
    for ann_id in coco.anns.keys():
        caption = coco.anns[ann_id]['caption']
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        for token in tokens:
            counter[token]+=1

    # create vocab
    vocab = Vocabulary(threshold)
    for word, count in counter.items():
        if count>threshold:
            vocab.add_word(word)

    return vocab

def save_vocab(vocab, vocab_file):
    """Save the vocabulary object to a pickle file."""
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary saved to {vocab_file}")


def main(args):
    ann_file = args.caption_path  # Path to COCO annotation file
    vocab_file = 'vocab.pkl'  # Output file to save the vocabulary

    if os.path.exists(vocab_file):
        print(f"Vocabulary file {vocab_file} already exists!")
    else:
        # Build the vocabulary from the COCO captions
        print("Building vocabulary...")
        vocab = build_vocab(ann_file, threshold=5)

        # Save the vocabulary to a pickle file
        save_vocab(vocab, vocab_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vocab building')
    parser.add_argument('--caption_path', type=str, default='data/captions_train2017.json', help="path caption jason file")
    args = parser.parse_args()
    main(args)
    
