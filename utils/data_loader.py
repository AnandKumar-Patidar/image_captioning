from torch.utils.data import DataLoader

def collate_fn(data, pad_idx):
    """Creates mini-batch tensors from the list of tuples (image, caption)."""
    # Sort data by caption length (descending)
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Stack images into a single tensor
    images = torch.stack(images, 0)

    # Pad the captions to the maximum length in the batch
    lengths = [len(cap) for cap in captions]
    padded_captions = pad_sequence(captions, batch_first=True, padding_value=pad_idx)

    return images, padded_captions, lengths
def get_loader(root, ann_file, vocab, transform, batch_size, shuffle=True, num_workers=4, subset_fraction=1.0):
    """Returns DataLoader for custom COCO dataset.
    
    Args:
        root (string): Directory containing the images.
        ann_file (string): Path to COCO annotation file.
        vocab (Vocabulary): Vocabulary object.
        transform (callable, optional): A function/transform to apply to images.
        batch_size (int): How many samples per batch.
        shuffle (bool, optional): Whether to shuffle the data at every epoch.
        num_workers (int, optional): Number of subprocesses to use for data loading.
        subset_fraction (float, optional): Fraction of dataset to use.
    
    Returns:
        data_loader (DataLoader): Dataloader for COCO dataset.
    """
    dataset = CocoDataset(root=root, ann_file=ann_file, vocab=vocab, transform=transform, subset_fraction=subset_fraction)
    pad_idx = vocab('<pad>')

    # Create and return the DataLoader
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             collate_fn=lambda x: collate_fn(x, pad_idx))
    return data_loader
