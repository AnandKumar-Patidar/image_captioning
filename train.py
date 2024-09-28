import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from models.encoder import EncoderCNN
from models.decoder import DecoderRNN
import torchvision.transforms as transforms
from utils import data_loader 
from utils import vocab as vc
import os


image_root = 'data'
ann_file = 'data/captions_train2017.json'
save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# Build vocabulary (done earlier)
vocab = vc.build_vocab(ann_file, threshold=5)

# Hyperparameters
embed_size = 256
hidden_size = 512
vocab_size = len(vocab)  # Vocabulary size (make sure vocab.__len__() is implemented)
num_epochs = 1
learning_rate = 0.001
log_interval = 1  # Log every 10 batches
batch_size = 32  # Batch size for training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
subset_fraction = 0.2
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
        
        # Forward pass: Extract image features using the encoder
        features = encoder(images)  # (batch_size, embed_size)

        # Forward pass: Generate captions using the decoder
        outputs = decoder(features, captions)  # (batch_size, max_caption_length, vocab_size)
         # Trim the outputs to exclude the prediction for <start>
        outputs = outputs[:, :-1, :]  # (batch_size, max_caption_length - 1, vocab_size)

        # Flatten the outputs for loss calculation
        outputs = outputs.contiguous().view(-1, vocab_size)
        
        # Shift the target captions to exclude the <start> token
        targets = captions[:, 1:].contiguous().view(-1) 

        # Compute the loss
        loss = criterion(outputs, targets)
        # Backpropagation
        optimizer.zero_grad()
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
    
    # Save model checkpoints at the end of each epoch
    torch.save(encoder.state_dict(), os.path.join(save_dir, f'encoder_cnn_{subset_fraction}_epoch_{epoch+1}_loss_{avg_loss:.4f}.pth'))
    torch.save(decoder.state_dict(), os.path.join(save_dir, f'decoder_rnn_{subset_fraction}_epoch_{epoch+1}_loss_{avg_loss:.4f}.pth'))
