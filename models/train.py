import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models

# Hyperparameters
embed_size = 256
hidden_size = 512
vocab_size = len(vocab)  # Vocabulary size (make sure vocab.__len__() is implemented)
num_epochs = 10
learning_rate = 0.001
log_interval = 10  # Log every 10 batches
batch_size = 32  # Batch size for training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Initialize models
encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers=1).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

# Load data
# Assume you have already created a DataLoader called data_loader
# data_loader = DataLoader(...)

# Training loop
for epoch in range(num_epochs):
    encoder.train()  # Set encoder to training mode
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
        
        # Adjust the outputs to exclude the last time step, to match the target captions[:, 1:]
        outputs = outputs[:, :-1, :]  # Exclude the last predicted word

# Flatten the outputs and the target captions
        loss = criterion(outputs.contiguous().view(-1, vocab_size), captions[:, 1:].contiguous().view(-1))

        
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
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {total_loss / len(data_loader):.4f}")
