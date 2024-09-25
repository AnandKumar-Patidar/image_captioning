
import torch.nn as nn
import torch

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        # Embedding layer: converts word indices into dense vectors of size embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # LSTM: input to hidden, hidden_size must match the size of features from CNN
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer to map LSTM output to vocab_size
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Initialize the hidden state (if needed)
        self.hidden_size = hidden_size
        
        # Optional dropout to prevent overfitting
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, features, captions):
        """
        Forward pass of the decoder.
        Arguments:
        - features: Tensor of shape (batch_size, feature_size=512)
        - captions: Tensor of shape (batch_size, max_caption_length), word indices
        
        Returns:
        - outputs: Tensor of shape (batch_size, max_caption_length, vocab_size), word predictions
        """
        
        # Embedding the captions, excluding the <end> token"
        embeddings = self.embedding(captions[:, :-1])
        
        # Concatenate the features with the embedded captions
        # Features are passed as input to the first time step
        features = features.unsqueeze(1)  # shape (batch_size, 1, feature_size)
        lstm_input = torch.cat((features, embeddings), 1)  # shape (batch_size, 1 + caption_length, embed_size)
        
        # Pass the concatenated inputs through the LSTM
        lstm_out, _ = self.lstm(lstm_input)
        
        # Pass the LSTM output through the fully connected layer to get word predictions
        outputs = self.fc(lstm_out)
        
        return outputs
