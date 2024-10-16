import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from models.encoder import EncoderCNN
from models.decoder import DecoderRNN
import torchvision.transforms as transforms
from utils import data_loader 
from utils import vocab as vc
from utils.vocab import Vocabulary
from datetime import datetime

import os
import pickle

image_root = 'data'
ann_file = 'data/captions_train2017.json'
save_dir = 'checkpoints/'
vocab_file = 'vocab.pkl'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Build vocabulary (done earlier)
if os.path.exists(vocab_file):
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
else:
    vocab = vc.build_vocab(ann_file, threshold=5)
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f)

# Hyperparameters
embed_size = 256
hidden_size = 512
vocab_size = len(vocab)  # Vocabulary size (make sure vocab.__len__() is implemented)
num_epochs = 20
learning_rate = 0.001
log_interval = 1  # Log every 10 batches
batch_size = 32  # Batch size for training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
subset_fraction = 0.1
# Initialize models
encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers=1).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)


# Load data
# Assume you have already created a DataLoader called data_loader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
data_loader = data_loader.get_loader(image_root, ann_file, vocab, transform, batch_size = 32, shuffle=True, num_workers=4, subset_fraction=subset_fraction)

# Training loop
for epoch in range(num_epochs):
    # encoder.train()  # Set encoder to training mode
    decoder.train()  # Set decoder to training mode
    
    total_loss = 0  # Keep track of loss for the entire epoch
    
    for i, (images, captions, lengths) in enumerate(data_loader):
        
        # Move images and captions to the device (GPU or CPU)
        images = images.to(device)
        captions = captions.to(device)
        optimizer.zero_grad()
        
        # Forward pass: Extract image features using the encoder
        features = encoder(images)  # (batch_size, embed_size)

        # Forward pass: Generate captions using the decoder
        outputs = decoder(features, captions)  # (batch_size, max_caption_length, vocab_size)
        outputs = outputs.view(-1, vocab_size)
        _, predicted_indices = torch.max(outputs, dim=1)

        # Flatten the outputs for loss calculation
        # outputs = outputs.contiguous().view(-1, vocab_size)
        
        # Shift the target captions to exclude the <start> token
        targets = captions[:,:].contiguous().view(-1) 
        print(f"output shape: {outputs.shape}, predcited indices shape: {predicted_indices.shape},target shape: {targets.shape}")
        for i in range(len(targets)):
            predicted_words = [vocab.idx2word[predicted_indices[i].item()] ]
            target_words = [vocab.idx2word[targets[i].item()]]
            print(f"target: {target_words}, predicted words: {predicted_words}")
        # break
        # Compute the loss
        loss = criterion(outputs, targets)
        # Backpropagation
        loss.backward()
        
        # Update the model's parameters
        optimizer.step()

        total_loss += loss.item()
        
        # Logging the loss every log_interval
        if i % log_interval == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}")
    
    # Average loss for the epoch
    # Calculate average loss for the epoch
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

def save_checkpoint(encoder, decoder, save_dir, epoch, avg_loss, subset_fraction):
    # Get the current date and time
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Save encoder and decoder models with date and time
    encoder_filename = f'encoder_cnn_{subset_fraction}_epoch_{epoch+1}_loss_{avg_loss:.4f}_{current_time}.pth'
    decoder_filename = f'decoder_rnn_{subset_fraction}_epoch_{epoch+1}_loss_{avg_loss:.4f}_{current_time}.pth'
    
    torch.save(encoder.state_dict(), os.path.join(save_dir, encoder_filename))
    torch.save(decoder.state_dict(), os.path.join(save_dir, decoder_filename))

save_checkpoint(encoder, decoder, save_dir, epoch, avg_loss, subset_fraction)
