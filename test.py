import torch
from torchvision import transforms
from models.encoder import EncoderCNN
from models.decoder import DecoderRNN
from utils.vocab import Vocabulary
from PIL import Image
import argparse
from utils import vocab as vc
import os
import pickle
import torch.nn.functional as F

ann_file = 'data/captions_train2017.json'
encoder_path = '/home/mirsee/image_captioning/checkpoints/encoder_cnn_0.01_epoch_1_loss_3.3651.pth'
decoder_path = '/home/mirsee/image_captioning/checkpoints/decoder_rnn_0.01_epoch_1_loss_3.3651.pth'
image_path = 'bike.jpg'
vocab_file = 'vocab.pkl'

if os.path.exists(vocab_file):
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
else:
    vocab = vc.build_vocab(ann_file, threshold=5)
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f)

def load_image(image_path, transfrom = None):
    image = Image.open(image_path).convert("RGB")
    if transfrom:
        image = transfrom(image).unsqueeze(0)
    return image

def generate_caption(encoder, decoder, vocab, image_path, device, max_length=20, temperature=1.0):
    """
    Generate caption using trained model with softmax and sampling.
    """
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = load_image(image_path, transform).to(device)

    # Encode the image using the trained encoder
    with torch.no_grad():
        features = encoder(image)

    caption = ['<start>']
    input_word = torch.tensor([vocab('<start>')]).unsqueeze(0).to(device)

    # Generate caption word by word using softmax sampling
    for _ in range(max_length):
        with torch.no_grad():
            output = decoder(features, input_word)

            # Apply softmax with temperature
            output = output / temperature
            probabilities = F.softmax(output, dim=-1)
            
            # Sample from the distribution instead of greedy decoding
            predicted_word_idx = torch.multinomial(probabilities.squeeze(0), 1).item()
            word = vocab.idx2word[predicted_word_idx]
            caption.append(word)

            # If <end> token is generated, stop prediction
            if word == '<end>':
                break

            # Prepare the next input word
            input_word = torch.tensor([predicted_word_idx]).unsqueeze(0).to(device)

    # Return caption ignoring <start> and <end> tokens
    return ' '.join(caption[1:-1])   
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # encoder = EncoderCNN(args.embed_size).to(device)
    # deocder = DecoderRNN(args.embed_size, args.hidden_size, args.vocab_size, num_layers = 1).to(device)

    # encoder.load_state_dict(torch.load(args.encoder_path, map_location = device))
    # decoder.load_state_dict(torch.load(args.decoder_path, map_location = device))

    # vocab = Vocabulary(args.vocab_path)
    encoder = EncoderCNN(256).to(device)
    decoder = DecoderRNN(256, 512, len(vocab), num_layers = 1).to(device)

    encoder.load_state_dict(torch.load(encoder_path, map_location = device, weights_only=True))
    encoder.eval()
    decoder.load_state_dict(torch.load(decoder_path, map_location = device, weights_only=True))
    decoder.eval()


    caption = generate_caption(encoder, decoder, vocab,image_path, device)
    print(f"Generated caption: {caption}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Test the image captioning model.')
    # parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    # parser.add_argument('--encoder_path', type=str, required=True, help='Path to the trained encoder model.')
    # parser.add_argument('--decoder_path', type=str, required=True, help='Path to the trained decoder model.')
    # parser.add_argument('--vocab_path', type=str, required=True, help='Path to the vocabulary object.')
    # parser.add_argument('--embed_size', type=int, default=256, help='Embedding size.')
    # parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size of the LSTM.')
    # parser.add_argument('--vocab_size', type=int, required=True, help='Size of the vocabulary.')

    # args = parser.parse_args()
    # main(args)
    main()


    
