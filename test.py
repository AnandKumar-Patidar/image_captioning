from utils import vocab as vc

image_root = 'data/'
ann_file = 'data/captions_train2017.json'
# Build vocabulary (done earlier)
vocab = vc.build_vocab(ann_file, threshold=5)