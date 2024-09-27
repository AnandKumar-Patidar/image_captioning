from pycocotools.coco import COCO
import requests
from io import BytesIO
from PIL import Image
import nltk
from torch.utils.data import Dataset
import random
import torch




class CocoDataset(Dataset):
    def __init__(self, root, ann_file, vocab, transform=None, subset_fraction=1.0):
        """
        Args:
            root (string): Directory with all the images.
            ann_file (string): Path to the annotation file.
            vocab (Vocabulary): Vocabulary object for tokenizing captions.
            transform (callable, optional): Transform to be applied to the images.
            subset_fraction (float, optional): Fraction of dataset to use, e.g., 0.25 for 1/4th of the data.
        """
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.anns.keys())  # List of annotation IDs

        # Use only a subset of the dataset if subset_fraction is less than 1.0
        if subset_fraction < 1.0:
            subset_size = int(len(self.ids) * subset_fraction)
            self.ids = random.sample(self.ids, subset_size)

        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        img_url = img_info['coco_url']  # Get the image URL from COCO annotations

        # Fetch the image from the URL
        response = requests.get(img_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')


        # Load the image

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Tokenize the caption and convert to indices
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption_idx = [self.vocab('<start>')] + [self.vocab(token) for token in tokens] + [self.vocab('<end>')]
        caption_tensor = torch.LongTensor(caption_idx)

        return image, caption_tensor
